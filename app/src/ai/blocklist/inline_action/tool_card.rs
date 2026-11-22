//! Unified tool card rendering helper, aligned with opencode TUI's `InlineTool` / `BlockTool`.
//!
//! ## Design philosophy
//!
//! opencode renders every ToolPart strictly with a 4-state machine:
//! - `pending` (args still accumulating): light gray text "Writing command..." / "Reading file..."
//! - `running` (args complete, executing): BrailleSpinner + title text
//! - `completed` (finished successfully): static icon + tool description, collapsible
//! - `error` (failed / denied): red error text, whole text STRIKETHROUGH when denied
//!
//! All 12 built-in tools (Bash/Read/Glob/Grep/Edit/Write/...) use only the InlineTool /
//! BlockTool components; a new tool **only supplies semantics** and does not reimplement the card skeleton.
//!
//! ## Current state in warp
//!
//! In warp's inline_action/ directory, each view (web_search.rs / web_fetch.rs /
//! requested_command.rs / requested_action.rs / ...) renders its own
//! complete card (header + body + footer + permission ring + state switching),
//! with at least ~150 lines of repeated boilerplate. This is historical baggage, and **a full refactor would need to change 12+ views at once**,
//! which is risky and meets resistance.
//!
//! This module is the **entry point for an incremental refactor**:
//! 1. Define a unified API ([`ToolCardState`] state machine + [`ToolCardSpec`] builder);
//! 2. Provide two helpers, [`render_inline_tool_card`] / [`render_block_tool_card`];
//! 3. **New inline_actions should prefer this module**; old views stay untouched until a separate PR converges them.
//!
//! `render_loading_header_animated` / `render_terminal_header_strikethrough` have already been added in
//! `search_results_common.rs`; this module layers the full spec abstraction on top of them.

use warp_core::ui::appearance::Appearance;
use warp_core::ui::theme::Fill;
use warpui::elements::shimmering_text::ShimmeringTextStateHandle;
use warpui::elements::{
    ConstrainedBox, Container, CornerRadius, CrossAxisAlignment, Element, Flex, MainAxisAlignment,
    ParentElement, Radius, Shrinkable,
};
use warpui::{AppContext, SingletonEntity};

use super::inline_action_header::{
    ICON_MARGIN, INLINE_ACTION_HEADER_VERTICAL_PADDING, INLINE_ACTION_HORIZONTAL_PADDING,
};
use super::inline_action_icons::icon_size;
use crate::ui_components::spinner::SpinnerStateHandle;

/// Current state of a tool card. **Strictly 5 states, aligned with opencode TUI**:
/// do not add intermediate states for convenience; every render branch accepts only these 5 cases.
///
/// 5 states rather than opencode's 4: the extra one is [`Self::PermissionPending`], which maps to warp's
/// `AIActionStatus::Blocked` (waiting for user approval). opencode folds this into InlineTool's
/// whole-card fg→warning color logic; we make it an explicit case for clarity.
#[derive(Clone)]
pub enum ToolCardState {
    /// args still accumulating or the tool has not actually started. Visual: static icon + a progressive phrase such as "Writing command..."
    /// + light gray text.
    Pending {
        /// Progressive phrase, e.g. "Writing command", "Reading file". No trailing `...` needed;
        /// it is appended automatically when rendering.
        verb: String,
    },
    /// Tool is executing. Visual: `BrailleSpinner` (80ms frame switching) + ShimmeringText title.
    Running {
        title: String,
        spinner_handle: SpinnerStateHandle,
        shimmer_handle: ShimmeringTextStateHandle,
    },
    /// Waiting for the user to allow execution (`AIActionStatus::Blocked`).
    /// Visual: **header background switches to warning yellow**, text keeps high contrast,
    /// aligned with opencode's `if (permission()) return theme.warning`.
    /// detail is usually "OK if I run this command?" / "OK if I call this MCP tool?".
    PermissionPending { title: String, detail: String },
    /// Tool completed successfully. Visual: green check icon + tool description.
    Completed { title: String },
    /// Tool failed / user denied. With `denied=true` the title text gets a STRIKETHROUGH
    /// to express "rejected", aligned with opencode `<text attributes={STRIKETHROUGH}>`.
    Error {
        title: String,
        denied: bool,
        detail: Option<String>,
    },
}

impl ToolCardState {
    /// Equivalent to opencode `part.state.status === "running"`. The spinner is shown only while Running.
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running { .. })
    }

    /// Equivalent to opencode `part.state.status === "completed"`. Can be hidden by the
    /// hide_completed_tool_cards setting.
    pub fn is_completed(&self) -> bool {
        matches!(self, Self::Completed { .. })
    }

    /// Whether denied (user rejected), used to switch to the strikethrough visual.
    pub fn is_denied(&self) -> bool {
        matches!(self, Self::Error { denied: true, .. })
    }

    /// Whether permission is pending (waiting for user approval), used to switch to the warning background color.
    pub fn is_permission_pending(&self) -> bool {
        matches!(self, Self::PermissionPending { .. })
    }
}

/// Tool card spec: all the information the caller must supply.
pub struct ToolCardSpec {
    /// Tool icon (used in terminal states; Pending/Running pick a spinner on their own based on state).
    pub icon: warpui::elements::Icon,
    /// Current state.
    pub state: ToolCardState,
}

/// Render an inline-mode tool card (single-line icon + text). Aligned with opencode `InlineTool`.
///
/// Suited to short descriptions: Glob "*.rs" / Grep "TODO" / WebFetch URL.
/// **Limitation**: body height is always 1 line; complex content (diff / file list) goes through [`render_block_tool_card`].
pub fn render_inline_tool_card(spec: ToolCardSpec, app: &AppContext) -> Box<dyn Element> {
    let appearance = Appearance::as_ref(app);
    let theme = appearance.theme();
    // T3-6: permission pending uses the warning yellow background; others use the default surface_2 background.
    let header_background: Fill = if spec.state.is_permission_pending() {
        Fill::Solid(theme.ui_warning_color())
    } else {
        theme.surface_2()
    };

    let mut row = Flex::row()
        .with_main_axis_alignment(MainAxisAlignment::Start)
        .with_cross_axis_alignment(CrossAxisAlignment::Center);

    // icon: swap in BrailleSpinner while Running; other states use the passed-in icon.
    let icon_element: Box<dyn Element> = match &spec.state {
        ToolCardState::Running { spinner_handle, .. } => {
            use warp_core::ui::theme::AnsiColorIdentifier;
            let color = AnsiColorIdentifier::Yellow.to_ansi_color(&theme.terminal_colors().normal);
            Box::new(crate::ui_components::spinner::BrailleSpinner::new(
                appearance.ui_font_family(),
                appearance.monospace_font_size(),
                color,
                spinner_handle.clone(),
            ))
        }
        _ => spec.icon.finish(),
    };
    let icon_box = ConstrainedBox::new(icon_element)
        .with_width(icon_size(app))
        .with_height(icon_size(app))
        .finish();
    row.add_child(
        Container::new(icon_box)
            .with_margin_right(ICON_MARGIN)
            .finish(),
    );

    // Text: each of the states builds its own.
    let title_element = build_title_text(&spec.state, header_background, app);
    row.add_child(Shrinkable::new(1.0, title_element).finish());

    Container::new(row.finish())
        .with_horizontal_padding(INLINE_ACTION_HORIZONTAL_PADDING)
        .with_vertical_padding(INLINE_ACTION_HEADER_VERTICAL_PADDING)
        .with_background(header_background)
        .with_corner_radius(CornerRadius::with_all(Radius::Pixels(8.)))
        .finish()
}

/// Render a block-mode tool card (header + body). Aligned with opencode `BlockTool`.
///
/// The header is the same as inline_tool_card; the body is any Element passed in by the caller (diff, file list,
/// output preview, etc.). While Running the header shows the spinner and the body is usually in-progress data.
pub fn render_block_tool_card(
    spec: ToolCardSpec,
    body: Box<dyn Element>,
    app: &AppContext,
) -> Box<dyn Element> {
    let appearance = Appearance::as_ref(app);
    let theme = appearance.theme();
    let body_background = theme.surface_1();

    let header = render_inline_tool_card(spec, app);
    let body_container = Container::new(body)
        .with_background(body_background)
        .with_horizontal_padding(INLINE_ACTION_HORIZONTAL_PADDING)
        .with_vertical_padding(INLINE_ACTION_HEADER_VERTICAL_PADDING)
        .with_corner_radius(CornerRadius::with_bottom(Radius::Pixels(8.)))
        .finish();

    let mut col = Flex::column().with_cross_axis_alignment(CrossAxisAlignment::Stretch);
    col.add_child(header);
    col.add_child(body_container);
    col.finish()
}

fn build_title_text(
    state: &ToolCardState,
    header_background: Fill,
    app: &AppContext,
) -> Box<dyn Element> {
    use warpui::elements::shimmering_text::{ShimmerConfig, ShimmeringTextElement};
    use warpui::elements::Text;

    let appearance = Appearance::as_ref(app);
    let theme = appearance.theme();

    match state {
        ToolCardState::Pending { verb } => {
            let color = theme.sub_text_color(header_background).into_solid();
            Text::new_inline(
                format!("{verb}..."),
                appearance.ui_font_family(),
                appearance.monospace_font_size(),
            )
            .with_color(color)
            .finish()
        }
        ToolCardState::Running {
            title,
            shimmer_handle,
            ..
        } => {
            let base_color = theme.sub_text_color(header_background).into_solid();
            let shimmer_color = theme.main_text_color(header_background).into_solid();
            ShimmeringTextElement::new(
                title.clone(),
                appearance.ui_font_family(),
                appearance.monospace_font_size(),
                base_color,
                shimmer_color,
                ShimmerConfig::default(),
                shimmer_handle.clone(),
            )
            .finish()
        }
        ToolCardState::Completed { title } => {
            let color = theme.main_text_color(header_background).into();
            Text::new_inline(
                title.clone(),
                appearance.ui_font_family(),
                appearance.monospace_font_size(),
            )
            .with_color(color)
            .finish()
        }
        ToolCardState::PermissionPending { title, detail } => {
            // Main title + detail sub-line. The background is already warning, so the text uses the main color for contrast.
            let main_color = theme.main_text_color(header_background).into();
            let detail_color = theme.sub_text_color(header_background).into_solid();
            let mut col = Flex::column().with_cross_axis_alignment(CrossAxisAlignment::Start);
            col.add_child(
                Text::new_inline(
                    title.clone(),
                    appearance.ui_font_family(),
                    appearance.monospace_font_size(),
                )
                .with_color(main_color)
                .finish(),
            );
            col.add_child(
                Text::new_inline(
                    detail.clone(),
                    appearance.ui_font_family(),
                    (appearance.monospace_font_size() - 1.).max(10.),
                )
                .with_color(detail_color)
                .finish(),
            );
            col.finish()
        }
        ToolCardState::Error {
            title,
            denied,
            detail,
        } => {
            use warpui::elements::{Highlight, HighlightedRange};
            use warpui::text_layout::TextStyle;

            // Main text: STRIKETHROUGH when denied; error does not get it but uses the sub color + a detail sub-line.
            let text_color = theme.sub_text_color(header_background).into_solid();
            let mut text_widget = Text::new_inline(
                title.clone(),
                appearance.ui_font_family(),
                appearance.monospace_font_size(),
            )
            .with_color(text_color);

            if *denied {
                let strike_style = TextStyle::new()
                    .with_show_strikethrough(true)
                    .with_foreground_color(text_color);
                let highlight = Highlight::default().with_text_style(strike_style);
                let len = title.chars().count();
                text_widget = text_widget.with_highlights(vec![HighlightedRange {
                    highlight,
                    highlight_indices: (0..len).collect(),
                }]);
            }

            // detail line: if present, join it in a column; otherwise just one line.
            if let Some(detail_text) = detail {
                let mut col = Flex::column().with_cross_axis_alignment(CrossAxisAlignment::Start);
                col.add_child(text_widget.finish());
                let detail_color = theme.ui_error_color();
                col.add_child(
                    Text::new_inline(
                        detail_text.clone(),
                        appearance.ui_font_family(),
                        (appearance.monospace_font_size() - 1.).max(10.),
                    )
                    .with_color(detail_color)
                    .finish(),
                );
                col.finish()
            } else {
                text_widget.finish()
            }
        }
    }
}
