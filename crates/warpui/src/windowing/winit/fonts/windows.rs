use super::{
    font_handle::FontHandle, FontFamily, LoadedSystemFonts, TextLayoutSystem,
    ValidateFontSupportsEn,
};
use crate::fonts::FontId;
use anyhow::Result;
use font_kit::loader::Loader as _;
use font_kit::{
    family_name::FamilyName as FKFamilyName, properties::Properties as FKProperties,
    properties::Style as FKStyle, properties::Weight as FKWeight, source::SystemSource as FKSource,
};
use itertools::Itertools;
use owned_ttf_parser::OwnedFace;
use std::collections::HashMap;
use std::sync::Arc;

/// Returns the BCP-47 locale string used to bias DirectWrite Han glyph fallback.
/// Kept in sync with the current UI locale (set by `app::i18n` via `crate::set_ui_locale`).
fn current_fallback_locale() -> String {
    crate::current_ui_locale()
}

/// Windows symbol fonts that are used to render window control icons. We specifically do not do any
/// validation of these fonts (i.e. to check if the font contains english characters).
const SYMBOL_ICON_FONTS: &[&str] = &["Segoe Fluent Icons", "Segoe MDL2 Assets"];

pub(crate) mod loader {
    use crate::fonts::FontInfo;

    use super::*;

    pub fn load_all_system_fonts() -> LoadedSystemFonts {
        let source = font_kit::source::SystemSource::new();
        let fonts = match source.all_fonts() {
            Ok(fonts) => fonts,
            Err(err) => {
                log::warn!("unable to retrieve all fonts from DirectWrite source: {err:?}");
                return LoadedSystemFonts(vec![]);
            }
        };

        let mut family_map = HashMap::new();

        for font_handle in fonts.into_iter() {
            if let Ok(font) = font_handle.load() {
                let family_name = font.family_name();
                let is_monospace = font.is_monospace();

                if font.glyph_for_char('m').is_none() {
                    // Only allow the user to select fonts that have an English character set.
                    log::debug!("skipping family {family_name:?} because no 'm' glyph was found");
                    continue;
                }
                // Convert font_kit::Handle into UI framework-specific FontHandle.
                let font_handle = match font_handle {
                    font_kit::handle::Handle::Path { path, font_index } => {
                        FontHandle::new(path, font_index, is_monospace)
                    }
                    font_kit::handle::Handle::Memory { bytes, font_index } => {
                        let owned_face_result = match Arc::try_unwrap(bytes) {
                            // If we can ensure ownership of the bytes, create an OwnedFace without copying.
                            Ok(owned_bytes) => OwnedFace::from_vec(owned_bytes, font_index),
                            // If we can't get sole ownership, create on OwnedFace from a copy the bytes
                            // (created by .to_vec()).
                            Err(shared_bytes) => {
                                OwnedFace::from_vec(shared_bytes.to_vec(), font_index)
                            }
                        };
                        match owned_face_result {
                            Ok(typeface) => FontHandle::from(typeface),
                            Err(err) => {
                                // If we can't parse the typeface, skip it.
                                log::warn!(
                                    "unable to parse typeface from family {family_name}: {err:?}"
                                );
                                continue;
                            }
                        }
                    }
                };

                let (entry_info, entry_family) = family_map
                    .entry(family_name.clone())
                    .or_insert_with(move || {
                        (
                            FontInfo {
                                family_name: family_name.clone(),
                                is_monospace,
                            },
                            FontFamily {
                                name: family_name,
                                fonts: vec![],
                            },
                        )
                    });
                entry_info.is_monospace |= is_monospace;
                entry_family.fonts.push(font_handle);
            }
        }
        LoadedSystemFonts(family_map.into_values().collect_vec())
    }

    pub fn load_system_font(font_family: &str) -> Result<FontFamily> {
        let source = font_kit::source::SystemSource::new();
        let family = source.select_family_by_name(font_family)?;

        let validate_supports_en = if SYMBOL_ICON_FONTS.contains(&font_family) {
            ValidateFontSupportsEn::No
        } else {
            ValidateFontSupportsEn::Yes
        };

        Ok(FontFamily {
            name: font_family.to_string(),
            fonts: family
                .fonts()
                .iter()
                .flat_map(|font_kit_handle| {
                    load_font_from_handle(font_kit_handle, validate_supports_en)
                })
                .collect_vec(),
        })
    }
}

impl TextLayoutSystem {
    /// Given a specific character and FontID, find alternate system fonts that can
    /// render that character.
    pub fn get_fallback_fonts_for_character(
        &self,
        character: char,
        font_id: FontId,
    ) -> Result<Vec<FontId>> {
        // Retrieve the font's family name and properties from the font store.
        // First, find the font's fontdb ID.
        let &original_font_id =
            self.font_id_map
                .read()
                .get_by_left(&font_id)
                .ok_or(anyhow::format_err!(
                    "No left entry found for {font_id:?} in font_id_map"
                ))?;
        let (style, weight, family_name) = self.get_font_info_from_store(original_font_id)?;
        let source = FKSource::new();
        let style = match style {
            fontdb::Style::Normal => FKStyle::Normal,
            fontdb::Style::Italic => FKStyle::Italic,
            fontdb::Style::Oblique => FKStyle::Oblique,
        };
        let weight = FKWeight(weight.0 as f32);
        let properties = FKProperties {
            style,
            weight,
            stretch: Default::default(),
        };

        let font_handle = source
            .select_best_match(
                &[
                    FKFamilyName::Title(family_name.to_owned()),
                    FKFamilyName::Monospace,
                ],
                &properties,
            )
            .map_err(|err| anyhow::anyhow!("Didn't find {family_name} in fontdb: {err}"))?;

        // Load fallback fonts for the requested character.
        let loaded_font = font_handle.load().map_err(|err| {
            anyhow::anyhow!("Unable to load typeface from font_kit Handle: {err:?}")
        })?;

        let locale = current_fallback_locale();

        // Locale-first CJK system fonts: on English / dev-environment Windows, DirectWrite's IDWriteFontFallback
        // does not consult the locale to resolve Han glyph ambiguity and defaults to Microsoft YaHei, so a Japanese UI ends up with Simplified Chinese glyphs.
        // So for shared CJK Han characters, we prepend the system font matching the current locale before the DirectWrite fallback
        // (for example ja-* → Yu Gothic UI).
        let mut fallback_font_vec: Vec<FontId> = Vec::new();
        if crate::is_shared_cjk_han(character) {
            for family in preferred_cjk_families_for_locale(&locale) {
                if let Ok(fam) = source.select_family_by_name(family) {
                    for fk_handle in fam.fonts() {
                        if let Ok(handle) =
                            load_font_from_handle(fk_handle, ValidateFontSupportsEn::No)
                        {
                            if let Ok(id) = self.insert_font(handle) {
                                fallback_font_vec.push(id);
                            }
                        }
                    }
                    if !fallback_font_vec.is_empty() {
                        break;
                    }
                }
            }
        }

        let fallback_result = loaded_font.get_fallbacks(character.to_string().as_str(), &locale);

        // Convert each font-kit fallback `Font` into a UI framework `FontHandle` and load it into
        // fontdb. We deliberately avoid `font_kit::Font::handle()` here: its default impl reads
        // the full font file into an `Arc<Vec<u8>>` and returns a `Handle::Memory` with
        // `font_index` hard-coded to `0` (see the FIXME at font-kit/src/loader.rs:172), which
        // bypasses `TextLayoutSystem::insert_font`'s path-based dedup and loses TTC face indices.
        // Instead we reach through `NativeFont` to the underlying `IDWriteFontFace` and recover
        // the on-disk file path + real face index, the same way
        // `DirectWriteSource::create_handle_from_dwrite_font` does for enumerated system fonts.
        // This lets fontdb mmap the file lazily and lets `insert_font` dedup by `(path, index)`,
        // so the same fallback family is loaded at most once per process.
        fallback_font_vec.extend(fallback_result.fonts.into_iter().flat_map(|fallback_font| {
            let loaded_handle = fallback_font_path_handle(&fallback_font.font).or_else(|| {
                // Last-resort fallback for fonts that aren't backed by a local file (e.g.
                // custom collection loaders). These don't appear in practice for DirectWrite
                // system fallbacks, but preserve the original byte-copy behavior so we
                // degrade gracefully instead of dropping the glyph.
                let handle = fallback_font.font.handle()?;
                load_font_from_handle(&handle, ValidateFontSupportsEn::No).ok()
            })?;
            self.insert_font(loaded_handle).ok()
        }));

        Ok(fallback_font_vec)
    }

    /// Warms up, at startup, the CJK font families preferred by the current UI locale (`preferred_cjk_families_for_locale`).
    /// Called once synchronously right after `FontDB` is constructed.
    ///
    /// Fixes the regression in zerx-lab/warp#68 "after startup, CJK fonts render wrongly until the panel is closed and reopened":
    /// PR #62 prepends locale-specific system CJK fonts to cosmic-text's fallback chain in `get_fallback_fonts_for_character`;
    /// but when the first CJK fallback fires on the first frame, `SystemSource::select_family_by_name`
    /// occasionally fails to find the font on the Windows DirectWrite cold path, so the prepended part is empty and the fallback lands on
    /// the cold output of `IDWriteFontFallback::MapCharacters` (which may be a family that is not the locale preference).
    /// Once that result is written into cosmic-text's `font_codepoint_support_info_cache` /
    /// `shape_run_cache` (per FontSystem instance, not invalidated when the locale stays the same), later renders keep reusing the wrong fallback
    /// until the panel is destroyed or the font size / font_id changes and bypasses the cache key.
    ///
    /// Warm-up synchronously loads the preferred families into fontdb here (`insert_font` dedupes via
    /// `loaded_fonts` by `(path, index)`, so a later `get_fallback_fonts_for_character` hit
    /// returns the existing `FontId` directly without loading again), removing the cold-path nondeterminism.
    ///
    /// Cost: a one-time `SystemSource` construction at startup, plus select, load and insert of one
    /// preferred family. `load_font_from_handle` converts a font_kit Path handle to an `OwnedFace`,
    /// and fontdb mmaps lazily. Measured at only a few milliseconds on Windows 11 with the stock YaHei UI installed,
    /// and the net gain is positive: previously every CJK character cache miss in `get_fallback_fonts_for_character`
    /// built a new `SystemSource` and re-selected/re-loaded; after warm-up this path hits the already loaded FontId on the first frame.
    ///
    /// Non-CJK locales also warm up the Windows default Simplified Chinese UI font family, so ordinary `Text` elements such as
    /// Chinese file names under an English UI have usable Han glyphs on the first frame, without enumerating all system fonts.
    ///
    /// Failure (family not installed / handle load failed) only logs a warning and does not block startup; it then falls back to
    /// the DirectWrite default fallback.
    pub(crate) fn warm_up_preferred_cjk_families(&self) {
        let locale = current_fallback_locale();
        let families = preferred_cjk_families_for_locale(&locale);
        if families.is_empty() {
            return;
        }
        let source = FKSource::new();
        let mut warmed_any = false;
        for family in families {
            let Ok(fam) = source.select_family_by_name(family) else {
                // Family not installed (e.g. a clean Windows 11 may lack SimSun); keep trying the next one.
                continue;
            };
            let mut family_loaded = false;
            for fk_handle in fam.fonts() {
                match load_font_from_handle(fk_handle, ValidateFontSupportsEn::No) {
                    Ok(handle) => {
                        if self.insert_font(handle).is_ok() {
                            family_loaded = true;
                        }
                    }
                    Err(err) => {
                        log::debug!(
                            "warm_up_preferred_cjk_families: skipping a face of {family:?}: {err:?}"
                        );
                    }
                }
            }
            if family_loaded {
                warmed_any = true;
                // Matches the "stop at the first family that hits" behavior of `get_fallback_fonts_for_character`,
                // so warm-up does not exceed the font set the fallback would actually use.
                break;
            }
        }
        if !warmed_any {
            log::warn!(
                "warm_up_preferred_cjk_families: could not warm up any CJK family ({families:?}) for locale={locale:?}; first-frame CJK fallback will take the DirectWrite cold path"
            );
        }
    }

    /// Critical section for fetching the font style, weight and family name from fontdb.
    /// This function performs the minimum work required to fetch this information from
    /// fontdb to minimize the amount of time spent holding a read lock on the font store.
    fn get_font_info_from_store(
        &self,
        font_id: fontdb::ID,
    ) -> Result<(fontdb::Style, fontdb::Weight, String)> {
        let store_read_lock = self.font_store.read();
        let db_read = store_read_lock.db();
        let face = db_read.face(font_id).ok_or(anyhow::anyhow!(
            "Unable to retrieve font face from fontdb font_store"
        ))?;
        let style = face.style;
        let weight = face.weight;
        let Some(en_us_family_info) = face.families.first() else {
            return Err(anyhow::anyhow!("Font face doesn't have any family names"));
        };
        let (family_name, _) = en_us_family_info;
        // Clone the family name because it's protected by the font store's RWLock.
        Ok((style, weight, family_name.to_owned()))
    }
}

fn load_font_from_handle(
    font_handle: &font_kit::handle::Handle,
    validate_supports_en_charset: ValidateFontSupportsEn,
) -> Result<FontHandle> {
    let font = font_handle.load()?;
    let is_monospace = font.is_monospace();
    if matches!(validate_supports_en_charset, ValidateFontSupportsEn::Yes) {
        font.glyph_for_char('m').ok_or(anyhow::format_err!(
            "No 'm' glyph found for font {}",
            font.full_name()
        ))?;
    }
    match font_handle {
        font_kit::handle::Handle::Path { path, font_index } => {
            Ok(FontHandle::new(path, *font_index, is_monospace))
        }
        font_kit::handle::Handle::Memory { bytes, font_index } => {
            let typeface = OwnedFace::from_vec(bytes.to_vec(), *font_index)?;
            Ok(FontHandle::from(typeface))
        }
    }
}

/// Extracts the primary language subtag of a BCP-47 tag, already ASCII-lowercased.
/// For example `ja-jp` → `ja`, `zh-hant-tw` → `zh`, `kok-in` → `kok`.
/// Used to determine the primary language exactly, so prefix matching like `starts_with("ko")` does not
/// misclassify `kok-IN` (Konkani) as Korean or `zha-CN` (Zhuang) as Chinese.
fn primary_subtag(lower: &str) -> &str {
    lower.split(['-', '_']).next().unwrap_or("")
}

const SIMPLIFIED_CHINESE_CJK_FAMILIES: &[&str] =
    &["Microsoft YaHei UI", "Microsoft YaHei", "SimSun"];
const TRADITIONAL_CHINESE_CJK_FAMILIES: &[&str] = &[
    "Microsoft JhengHei UI",
    "Microsoft JhengHei",
    "PMingLiU",
    "MingLiU",
];
const JAPANESE_CJK_FAMILIES: &[&str] = &[
    "Yu Gothic UI",
    "Yu Gothic",
    "Meiryo UI",
    "Meiryo",
    "MS Gothic",
];
const KOREAN_CJK_FAMILIES: &[&str] = &["Malgun Gothic", "Gulim", "Dotum"];

/// Returns the Windows system CJK font families for a locale, in priority order.
/// Used to override DirectWrite's Han fallback, which ignores the locale.
///
/// Routing recognizes both BCP-47 region subtags (zh-TW / zh-HK / zh-MO) and script subtags
/// (zh-Hant / zh-Hans, optionally with a region such as zh-Hant-TW), so callers need not normalize the tag first.
/// Non-CJK locales use the Simplified Chinese font family as a stable fallback, so Chinese file names under an English UI do not miss glyphs on the first frame.
fn preferred_cjk_families_for_locale(locale: &str) -> &'static [&'static str] {
    let lower = locale.to_ascii_lowercase();
    match primary_subtag(&lower) {
        "ja" => JAPANESE_CJK_FAMILIES,
        "ko" => KOREAN_CJK_FAMILIES,
        "zh" if is_zh_traditional(&lower) => TRADITIONAL_CHINESE_CJK_FAMILIES,
        "zh" => SIMPLIFIED_CHINESE_CJK_FAMILIES,
        _ => SIMPLIFIED_CHINESE_CJK_FAMILIES,
    }
}

/// Whether `lower` (a BCP-47 tag already ASCII-lowercased) refers to Traditional Chinese.
/// Matches both the region form (zh-tw / zh-hk / zh-mo) and the script-subtag form
/// (zh-hant, zh-hant-tw, zh-foo-hant, etc.). Requires a hyphen boundary,
/// so accidental matches such as `zh-hansolo` are avoided.
fn is_zh_traditional(lower: &str) -> bool {
    if primary_subtag(lower) != "zh" {
        return false;
    }
    if lower.starts_with("zh-tw") || lower.starts_with("zh-hk") || lower.starts_with("zh-mo") {
        return true;
    }
    // Iterate over the hyphen-separated subtags after the primary tag.
    lower.split('-').skip(1).any(|sub| sub == "hant")
}

/// Builds a path-backed [`FontHandle`] for a font-kit DirectWrite `Font` by reaching through
/// [`font_kit::loaders::directwrite::NativeFont`] to the underlying `IDWriteFontFace`.
///
/// This mirrors what font-kit itself does for enumerated system fonts in
/// `DirectWriteSource::create_handle_from_dwrite_font` (font-kit/src/sources/directwrite.rs:103),
/// and is the reason we carry `dwrote` as a direct dependency: font-kit's generic
/// `Loader::handle()` default returns a `Handle::Memory` with a byte copy of the full file, which
/// we specifically need to avoid on the per-character fallback path.
///
/// Returns `None` when DirectWrite cannot produce a local file path for the font, i.e. the font
/// was loaded via a custom collection loader or backed only by an in-memory stream. For system
/// fallback fonts returned by `IDWriteFontFallback::MapCharacters` against the system font
/// collection, a path is always available.
fn fallback_font_path_handle(font: &font_kit::loaders::directwrite::Font) -> Option<FontHandle> {
    let native = font.native_font();
    let file = native.dwrite_font_face.files().ok()?.into_iter().next()?;
    let path = file.font_file_path().ok()?;
    let font_index = native.dwrite_font_face.get_index();
    Some(FontHandle::new(path, font_index, font.is_monospace()))
}

#[cfg(test)]
mod tests {
    use super::{
        preferred_cjk_families_for_locale, JAPANESE_CJK_FAMILIES, KOREAN_CJK_FAMILIES,
        SIMPLIFIED_CHINESE_CJK_FAMILIES, TRADITIONAL_CHINESE_CJK_FAMILIES,
    };

    #[test]
    fn preferred_cjk_families_defaults_to_simplified_chinese_for_non_cjk_locale() {
        assert_eq!(
            preferred_cjk_families_for_locale("en-US"),
            SIMPLIFIED_CHINESE_CJK_FAMILIES
        );
        assert_eq!(
            preferred_cjk_families_for_locale(""),
            SIMPLIFIED_CHINESE_CJK_FAMILIES
        );
    }

    #[test]
    fn preferred_cjk_families_respects_cjk_locale() {
        assert_eq!(
            preferred_cjk_families_for_locale("zh-CN"),
            SIMPLIFIED_CHINESE_CJK_FAMILIES
        );
        assert_eq!(
            preferred_cjk_families_for_locale("zh-Hans-US"),
            SIMPLIFIED_CHINESE_CJK_FAMILIES
        );
        assert_eq!(
            preferred_cjk_families_for_locale("zh-TW"),
            TRADITIONAL_CHINESE_CJK_FAMILIES
        );
        assert_eq!(
            preferred_cjk_families_for_locale("zh-Hant-HK"),
            TRADITIONAL_CHINESE_CJK_FAMILIES
        );
        assert_eq!(
            preferred_cjk_families_for_locale("ja-JP"),
            JAPANESE_CJK_FAMILIES
        );
        assert_eq!(
            preferred_cjk_families_for_locale("ko-KR"),
            KOREAN_CJK_FAMILIES
        );
    }
}
