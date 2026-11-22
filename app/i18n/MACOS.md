# Steps required when adding a new language on macOS

For macOS to localize correctly, when adding a new language (such as `zh-TW` or `ja`), you also need to:

1. **Update the macOS plist configuration**: add the new language code to the `CFBundleLocalizations` array in every plist that declares localizations:
   - `app/assets/resources/mac/CLI-Info.plist` — edit the `<array>` under `<key>CFBundleLocalizations</key>`
   - `app/src/bin/local.rs` — edit the plist XML inside the `embed_plist::embed_info_plist_bytes!` macro
   - `app/src/bin/oss.rs` — edit the plist XML inside the `embed_plist::embed_info_plist_bytes!` macro
   - Example: add `<string>zh-TW</string>` to the `<array>`
2. **Update the build script**: add the new language code to the `plutil -insert/replace CFBundleLocalizations` commands in the `script/update_plist` script
