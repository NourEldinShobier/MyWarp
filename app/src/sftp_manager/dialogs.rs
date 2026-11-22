//! Dialog rendering components
//!
//! Renders the delete-confirm, rename, new-folder and file-details dialogs.
//! author: logic
//! date: 2026-05-26

use std::path::PathBuf;

use warp_core::ui::appearance::Appearance;
use warp_core::ui::icons::Icon;
use warpui::elements::{
    Border, ChildView, Clipped, ConstrainedBox, Container, CornerRadius, CrossAxisAlignment, Dismiss, Flex,
    Hoverable, MainAxisSize, MainAxisAlignment, MouseStateHandle, ParentElement, Radius, SavePosition, Shrinkable, Text,
};
use warpui::platform::Cursor;
use warpui::Element;
use warpui::ViewHandle;

use crate::editor::EditorView;
use crate::sftp_manager::browser::SftpBrowserAction;
use crate::sftp_manager::types::{format_size, Dialog, FileEntry, TransferDirection};

/// Maximum dialog width
const DIALOG_MAX_WIDTH: f32 = 360.0;
/// Maximum dialog height
const DIALOG_MAX_HEIGHT: f32 = 500.0;
/// Dialog padding
const DIALOG_PADDING: f32 = 16.0;
/// Minimum button width
const BUTTON_MIN_WIDTH: f32 = 80.0;
/// Button height
const BUTTON_HEIGHT: f32 = 32.0;

/// Popup shell container
///
/// Provides a uniform background, corner radius, border and padding.
fn dialog_shell(content: Box<dyn Element>, appearance: &Appearance) -> Box<dyn Element> {
    let theme = appearance.theme();
    Container::new(content)
        .with_background(theme.surface_1())
        .with_corner_radius(CornerRadius::with_all(Radius::Pixels(8.0)))
        .with_border(Border::all(1.0).with_border_fill(theme.surface_3()))
        .with_uniform_padding(DIALOG_PADDING)
        .finish()
}

/// Renders the title row (title + close button)
///
/// The title is wrapped in Shrinkable so the width adapts; the X close button sits on the right.
fn render_title_bar(title: &str, appearance: &Appearance, close_btn_state: MouseStateHandle) -> Box<dyn Element> {
    let theme = appearance.theme();
    let text_color = theme.active_ui_text_color();
    let ui_font = appearance.ui_font_family();
    let ui_font_size = appearance.ui_font_size();

    let title_el = Shrinkable::new(
        1.0,
        Text::new(title.to_string(), ui_font, ui_font_size)
            .with_color(text_color.into())
            .finish(),
    )
    .finish();

    let close_btn = render_icon_close_button(appearance, close_btn_state);

    Flex::row()
        .with_main_axis_size(MainAxisSize::Max)
        .with_main_axis_alignment(MainAxisAlignment::SpaceBetween)
        .with_cross_axis_alignment(CrossAxisAlignment::Center)
        .with_child(title_el)
        .with_child(close_btn)
        .finish()
}

/// Renders the X-icon close button
fn render_icon_close_button(appearance: &Appearance, mouse_state: MouseStateHandle) -> Box<dyn Element> {
    let theme = appearance.theme();
    let icon_color = theme.sub_text_color(theme.background());
    let icon_el = ConstrainedBox::new(Icon::X.to_warpui_icon(icon_color).finish())
        .with_width(12.0)
        .with_height(12.0)
        .finish();
    Hoverable::new(mouse_state, move |_| {
        Container::new(icon_el)
            .with_padding_left(4.0)
            .with_padding_right(4.0)
            .with_padding_top(4.0)
            .with_padding_bottom(4.0)
            .finish()
    })
    .with_cursor(Cursor::PointingHand)
    .on_click(|ctx, _, _| {
        ctx.dispatch_typed_action(SftpBrowserAction::CloseDialog);
    })
    .finish()
}

/// Renders an action button
///
/// Uses the accent background when is_accent is true, otherwise the surface_2 background.
fn render_button(
    label: &str,
    is_accent: bool,
    appearance: &Appearance,
    action: SftpBrowserAction,
    mouse_state: MouseStateHandle,
    position_id: Option<&str>,
) -> Box<dyn Element> {
    let theme = appearance.theme();
    let ui_font = appearance.ui_font_family();
    let ui_font_size = appearance.ui_font_size();
    let bg = if is_accent {
        theme.accent()
    } else {
        theme.surface_2()
    };
    let text_color = if is_accent {
        theme.background()
    } else {
        theme.active_ui_text_color()
    };
    let label_owned = label.to_string();

    let btn_el = Hoverable::new(mouse_state, move |_| {
        let text_el = Text::new(label_owned.clone(), ui_font, ui_font_size)
            .with_color(text_color.into())
            .finish();
        let centered = Flex::row()
            .with_main_axis_alignment(MainAxisAlignment::Center)
            .with_cross_axis_alignment(CrossAxisAlignment::Center)
            .with_main_axis_size(MainAxisSize::Max)
            .with_child(text_el)
            .finish();
        Container::new(
            ConstrainedBox::new(centered)
                .with_width(BUTTON_MIN_WIDTH)
                .with_height(BUTTON_HEIGHT)
                .finish(),
        )
        .with_background(bg)
        .with_corner_radius(CornerRadius::with_all(Radius::Pixels(4.0)))
        .finish()
    })
    .with_cursor(Cursor::PointingHand)
    .on_click(move |ctx, _, _| {
        ctx.dispatch_typed_action(action.clone());
    })
    .finish();

    match position_id {
        Some(id) => SavePosition::new(btn_el, id).finish(),
        None => btn_el,
    }
}

/// Renders the cancel button
fn render_cancel_button(appearance: &Appearance, mouse_state: MouseStateHandle) -> Box<dyn Element> {
    render_button(&crate::t!("common-cancel"), false, appearance, SftpBrowserAction::CloseDialog, mouse_state, Some("sftp_btn:dialog_cancel"))
}

/// Renders a descriptive confirm dialog (title + description + confirm/cancel buttons)
///
/// Used for delete, move and overwrite confirmations.
fn render_confirm_dialog(
    title: &str,
    description: &str,
    confirm_label: &str,
    confirm_action: SftpBrowserAction,
    appearance: &Appearance,
    confirm_btn_state: MouseStateHandle,
    cancel_btn_state: MouseStateHandle,
    close_btn_state: MouseStateHandle,
) -> Box<dyn Element> {
    let theme = appearance.theme();
    let sub_color = theme.sub_text_color(theme.background());
    let ui_font = appearance.ui_font_family();
    let ui_font_size = appearance.ui_font_size();

    let title_bar = render_title_bar(title, appearance, close_btn_state);

    let desc_el = Shrinkable::new(
        1.0,
        Text::new(description.to_string(), ui_font, ui_font_size)
            .with_color(sub_color.into())
            .finish(),
    )
    .finish();

    let confirm_btn = render_button(confirm_label, true, appearance, confirm_action, confirm_btn_state, Some("sftp_btn:dialog_confirm"));
    let cancel_btn = render_cancel_button(appearance, cancel_btn_state);

    let buttons = Flex::row()
        .with_cross_axis_alignment(CrossAxisAlignment::Center)
        .with_main_axis_alignment(MainAxisAlignment::End)
        .with_spacing(8.0)
        .with_child(confirm_btn)
        .with_child(cancel_btn)
        .finish();

    let content = Flex::column()
        .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
        .with_spacing(12.0)
        .with_child(title_bar)
        .with_child(desc_el)
        .with_child(buttons)
        .finish();

    let dialog_body = ConstrainedBox::new(dialog_shell(content, appearance))
        .with_max_width(DIALOG_MAX_WIDTH)
        .with_max_height(DIALOG_MAX_HEIGHT)
        .finish();

    wrap_dismiss(dialog_body)
}

/// Wraps popup content in Dismiss + a centered container
fn wrap_dismiss(dialog_content: Box<dyn Element>) -> Box<dyn Element> {
    Dismiss::new(dialog_content)
        .prevent_interaction_with_other_elements()
        .on_dismiss(|ctx, _| {
            ctx.dispatch_typed_action(SftpBrowserAction::CloseDialog);
        })
        .finish()
}

/// Renders the delete-confirm dialog
fn render_delete_confirm(
    paths: &[PathBuf],
    appearance: &Appearance,
    confirm_btn_state: MouseStateHandle,
    cancel_btn_state: MouseStateHandle,
    close_btn_state: MouseStateHandle,
) -> Box<dyn Element> {
    let count = paths.len();
    let desc = if count == 1 {
        let name = paths[0]
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| paths[0].display().to_string());
        crate::t!("sftp-dialog-delete-one", name = name)
    } else {
        crate::t!("sftp-dialog-delete-many", count = count)
    };

    render_confirm_dialog(
        &crate::t!("sftp-dialog-delete-title"),
        &desc,
        &crate::t!("common-delete"),
        SftpBrowserAction::ConfirmDelete,
        appearance,
        confirm_btn_state,
        cancel_btn_state,
        close_btn_state,
    )
}

/// Renders the rename dialog
fn render_rename(
    original_name: &str,
    editor: &ViewHandle<EditorView>,
    appearance: &Appearance,
    confirm_btn_state: MouseStateHandle,
    cancel_btn_state: MouseStateHandle,
    close_btn_state: MouseStateHandle,
) -> Box<dyn Element> {
    let theme = appearance.theme();
    let sub_color = theme.sub_text_color(theme.background());
    let ui_font = appearance.ui_font_family();
    let ui_font_size = appearance.ui_font_size();

    // Title row
    let title_bar = render_title_bar(&crate::t!("common-rename"), appearance, close_btn_state);

    // Current name hint
    let hint = crate::t!("sftp-dialog-current-name", name = original_name);
    let hint_el = Shrinkable::new(
        1.0,
        Text::new(hint, ui_font, ui_font_size)
            .with_color(sub_color.into())
            .finish(),
    )
    .finish();

    // Editor: Shrinkable + Clipped prevent long file names from overflowing
    let editor_el = Container::new(
        Shrinkable::new(
            1.0,
            Clipped::new(ChildView::new(editor).finish()).finish(),
        )
        .finish(),
    )
    .with_padding_left(8.0)
    .with_padding_right(8.0)
    .with_padding_top(4.0)
    .with_padding_bottom(4.0)
    .with_corner_radius(CornerRadius::with_all(Radius::Pixels(4.0)))
    .with_background(theme.surface_2())
    .finish();

    // Buttons
    let confirm_btn = render_button(
        &crate::t!("common-ok"),
        true,
        appearance,
        SftpBrowserAction::ConfirmRename,
        confirm_btn_state,
        Some("sftp_btn:dialog_confirm"),
    );
    let cancel_btn = render_cancel_button(appearance, cancel_btn_state);

    let buttons = Flex::row()
        .with_cross_axis_alignment(CrossAxisAlignment::Center)
        .with_main_axis_alignment(MainAxisAlignment::End)
        .with_spacing(8.0)
        .with_child(confirm_btn)
        .with_child(cancel_btn)
        .finish();

    let content = Flex::column()
        .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
        .with_spacing(12.0)
        .with_child(title_bar)
        .with_child(hint_el)
        .with_child(editor_el)
        .with_child(buttons)
        .finish();

    let dialog_body = ConstrainedBox::new(dialog_shell(content, appearance))
        .with_max_width(DIALOG_MAX_WIDTH)
        .with_max_height(DIALOG_MAX_HEIGHT)
        .finish();

    wrap_dismiss(dialog_body)
}

/// Renders the new-folder dialog
fn render_create_folder(
    editor: &ViewHandle<EditorView>,
    appearance: &Appearance,
    confirm_btn_state: MouseStateHandle,
    cancel_btn_state: MouseStateHandle,
    close_btn_state: MouseStateHandle,
) -> Box<dyn Element> {
    let theme = appearance.theme();

    // Title row
    let title_bar = render_title_bar(&crate::t!("sftp-dialog-create-folder-title"), appearance, close_btn_state);

    // Editor
    let editor_el = Container::new(
        Shrinkable::new(
            1.0,
            Clipped::new(ChildView::new(editor).finish()).finish(),
        )
        .finish(),
    )
    .with_padding_left(8.0)
    .with_padding_right(8.0)
    .with_padding_top(4.0)
    .with_padding_bottom(4.0)
    .with_corner_radius(CornerRadius::with_all(Radius::Pixels(4.0)))
    .with_background(theme.surface_2())
    .finish();

    // Buttons
    let confirm_btn = render_button(
        &crate::t!("sftp-dialog-create"),
        true,
        appearance,
        SftpBrowserAction::ConfirmNewFolder,
        confirm_btn_state,
        Some("sftp_btn:dialog_confirm"),
    );
    let cancel_btn = render_cancel_button(appearance, cancel_btn_state);

    let buttons = Flex::row()
        .with_cross_axis_alignment(CrossAxisAlignment::Center)
        .with_main_axis_alignment(MainAxisAlignment::End)
        .with_spacing(8.0)
        .with_child(confirm_btn)
        .with_child(cancel_btn)
        .finish();

    let content = Flex::column()
        .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
        .with_spacing(12.0)
        .with_child(title_bar)
        .with_child(editor_el)
        .with_child(buttons)
        .finish();

    let dialog_body = ConstrainedBox::new(dialog_shell(content, appearance))
        .with_max_width(DIALOG_MAX_WIDTH)
        .with_max_height(DIALOG_MAX_HEIGHT)
        .finish();

    wrap_dismiss(dialog_body)
}

/// Renders a single property row (label + value)
fn detail_row(label: &str, value: &str, appearance: &Appearance) -> Box<dyn Element> {
    let theme = appearance.theme();
    let sub_color = theme.sub_text_color(theme.background());
    let text_color = theme.active_ui_text_color();
    let ui_font = appearance.ui_font_family();
    let ui_font_size = appearance.ui_font_size();

    let label_el = ConstrainedBox::new(
        Text::new(label.to_string(), ui_font, ui_font_size)
            .with_color(sub_color.into())
            .finish(),
    )
    .with_width(80.0)
    .finish();

    let value_el = Shrinkable::new(
        1.0,
        Text::new(value.to_string(), ui_font, ui_font_size)
            .with_color(text_color.into())
            .finish(),
    )
    .finish();

    Flex::row()
        .with_cross_axis_alignment(CrossAxisAlignment::Center)
        .with_spacing(8.0)
        .with_child(label_el)
        .with_child(value_el)
        .finish()
}

/// Renders the file-details dialog
fn render_file_details(
    entry: &FileEntry,
    appearance: &Appearance,
    cancel_btn_state: MouseStateHandle,
    close_btn_state: MouseStateHandle,
) -> Box<dyn Element> {
    // Title row
    let title_bar = render_title_bar(&crate::t!("sftp-dialog-details-title"), appearance, close_btn_state);

    // Type
    let type_str = match entry.file_type {
        crate::sftp_manager::types::FileEntryType::File => crate::t!("sftp-file-type-file"),
        crate::sftp_manager::types::FileEntryType::Directory => crate::t!("sftp-file-type-directory"),
        crate::sftp_manager::types::FileEntryType::Symlink => crate::t!("sftp-file-type-symlink"),
        crate::sftp_manager::types::FileEntryType::Other => crate::t!("sftp-file-type-other"),
    };

    // Build property rows
    let mut rows = Flex::column()
        .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
        .with_spacing(8.0);

    rows.add_child(detail_row(&crate::t!("sftp-dialog-details-type"), &type_str, appearance));
    rows.add_child(detail_row(&crate::t!("sftp-dialog-details-size"), &format_size(entry.size), appearance));
    let modified = entry.modified.as_deref().unwrap_or("--");
    rows.add_child(detail_row(&crate::t!("sftp-dialog-details-modified"), modified, appearance));
    let permissions = entry.permissions.as_deref().unwrap_or("--");
    rows.add_child(detail_row(&crate::t!("sftp-dialog-details-permissions"), permissions, appearance));
    rows.add_child(detail_row(&crate::t!("sftp-dialog-details-path"), &entry.path.display().to_string(), appearance));

    // Close button
    let close_btn = render_cancel_button(appearance, cancel_btn_state);

    let content = Flex::column()
        .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
        .with_spacing(12.0)
        .with_child(title_bar)
        .with_child(ConstrainedBox::new(rows.finish()).with_max_height(250.0).finish())
        .with_child(close_btn)
        .finish();

    let dialog_body = ConstrainedBox::new(dialog_shell(content, appearance))
        .with_max_width(DIALOG_MAX_WIDTH)
        .with_max_height(DIALOG_MAX_HEIGHT)
        .finish();

    wrap_dismiss(dialog_body)
}

/// Renders the move dialog
fn render_move_dialog(
    source: &PathBuf,
    target_dir: &PathBuf,
    appearance: &Appearance,
    confirm_btn_state: MouseStateHandle,
    cancel_btn_state: MouseStateHandle,
    close_btn_state: MouseStateHandle,
) -> Box<dyn Element> {
    let source_name = source
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let target_display = target_dir.display();
    let desc = crate::t!(
        "sftp-dialog-move-description",
        source = source_name,
        target = target_display.to_string()
    );

    render_confirm_dialog(
        &crate::t!("sftp-dialog-move-title"),
        &desc,
        &crate::t!("sftp-dialog-move-confirm"),
        SftpBrowserAction::ConfirmMove,
        appearance,
        confirm_btn_state,
        cancel_btn_state,
        close_btn_state,
    )
}

/// Renders the overwrite-confirm dialog
fn render_overwrite_confirm(
    _source: &PathBuf,
    target: &PathBuf,
    _file_size: u64,
    direction: TransferDirection,
    appearance: &Appearance,
    confirm_btn_state: MouseStateHandle,
    cancel_btn_state: MouseStateHandle,
    close_btn_state: MouseStateHandle,
) -> Box<dyn Element> {
    let target_name = target
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let desc = match direction {
        TransferDirection::Upload => crate::t!("sftp-dialog-overwrite-upload", name = target_name),
        TransferDirection::Download => crate::t!("sftp-dialog-overwrite-download", name = target_name),
    };

    render_confirm_dialog(
        &crate::t!("sftp-dialog-overwrite-title"),
        &desc,
        &crate::t!("sftp-dialog-overwrite-confirm"),
        SftpBrowserAction::ConfirmOverwrite,
        appearance,
        confirm_btn_state,
        cancel_btn_state,
        close_btn_state,
    )
}

/// Renders the dialog (main entry point)
///
/// Dispatches to the matching render function by dialog type.
pub fn render_dialog(
    dialog: &Dialog,
    rename_editor: &ViewHandle<EditorView>,
    new_folder_editor: &ViewHandle<EditorView>,
    appearance: &Appearance,
    confirm_btn_state: MouseStateHandle,
    cancel_btn_state: MouseStateHandle,
    close_btn_state: MouseStateHandle,
) -> Box<dyn Element> {
    match dialog {
        Dialog::DeleteConfirm { paths, .. } => {
            render_delete_confirm(paths, appearance, confirm_btn_state, cancel_btn_state, close_btn_state)
        }
        Dialog::Rename {
            original_name,
            ..
        } => render_rename(original_name, rename_editor, appearance, confirm_btn_state, cancel_btn_state, close_btn_state),
        Dialog::CreateFolder { .. } => {
            render_create_folder(new_folder_editor, appearance, confirm_btn_state, cancel_btn_state, close_btn_state)
        }
        Dialog::FileDetails { entry } => {
            render_file_details(entry, appearance, cancel_btn_state, close_btn_state)
        }
        Dialog::Move { source, target_dir } => {
            render_move_dialog(source, target_dir, appearance, confirm_btn_state, cancel_btn_state, close_btn_state)
        }
        Dialog::OverwriteConfirm { source, target, file_size, direction } => {
            render_overwrite_confirm(source, target, *file_size, *direction, appearance, confirm_btn_state, cancel_btn_state, close_btn_state)
        }
        Dialog::CloseTransferPanelConfirm => {
            render_confirm_dialog(
                &crate::t!("sftp-dialog-close-transfer-title"),
                &crate::t!("sftp-dialog-close-transfer-body"),
                &crate::t!("common-close"),
                SftpBrowserAction::ConfirmCloseTransferPanel,
                appearance,
                confirm_btn_state,
                cancel_btn_state,
                close_btn_state,
            )
        }
    }
}
