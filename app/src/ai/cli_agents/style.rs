//! Reply styling: turns `[[Done]]`-style tags into inline-code chips, line by
//! line as the text streams. Chips render in Warp's own inline-code style, so they follow the
//! user's theme. Code fences and inline code are left alone.

/// Buffers streamed text into whole lines so tags split across chunks are still found.
#[derive(Default)]
pub(crate) struct LineStyler {
    pending: String,
    in_fence: bool,
}

impl LineStyler {
    /// Adds streamed text; returns the styled text of every line completed so far.
    pub(crate) fn push(&mut self, text: &str) -> String {
        self.pending.push_str(text);
        let Some(end) = self.pending.rfind('\n') else { return String::new() };
        let complete: String = self.pending.drain(..=end).collect();
        complete.split_inclusive('\n').map(|l| self.line(l)).collect()
    }

    /// Styled text of the unfinished last line, if any.
    pub(crate) fn flush(&mut self) -> String {
        let rest = std::mem::take(&mut self.pending);
        self.line(&rest)
    }

    fn line(&mut self, line: &str) -> String {
        if line.trim_start().starts_with("```") || line.trim_start().starts_with("~~~") {
            self.in_fence = !self.in_fence;
            return line.to_owned();
        }
        if self.in_fence { line.to_owned() } else { style_line(line) }
    }
}

/// Replaces `[[Word]]` / `[[Word: detail]]` outside inline-code spans.
fn style_line(line: &str) -> String {
    line.split('`')
        .enumerate()
        // Odd segments sit between backticks, i.e. inside inline code.
        .map(|(i, seg)| if i % 2 == 1 { seg.to_owned() } else { chips(seg) })
        .collect::<Vec<_>>()
        .join("`")
}

fn chips(seg: &str) -> String {
    let mut out = String::new();
    let mut rest = seg;
    while let Some(start) = rest.find("[[") {
        let Some(len) = rest[start + 2..].find("]]") else { break };
        let inner = &rest[start + 2..start + 2 + len];
        out.push_str(&rest[..start]);
        match chip(inner) {
            Some(c) => out.push_str(&c),
            None => out.push_str(&rest[start..start + 4 + len]),
        }
        rest = &rest[start + 4 + len..];
    }
    out.push_str(rest);
    out
}

fn chip(inner: &str) -> Option<String> {
    let inner = inner.trim();
    let word = inner.split([':', ' ']).next().unwrap_or_default();
    if inner.is_empty() || inner.len() > 40 || inner.contains('`') || !word.chars().all(char::is_alphabetic) {
        return None;
    }
    Some(format!("`{inner}`"))
}

/// Appended to the CLI's system prompt so replies use the tags above.
pub(crate) const GUIDANCE: &str = "Reply style (display only): when a result needs a status label, write it as a tag in double square brackets, e.g. [[Done]], [[Failed]], [[Warning]], [[Next]], [[Passed 12/12]]. Use tags sparingly, only for real outcomes. Keep paragraphs short and prefer short lists over long prose.";

#[cfg(test)]
#[path = "style_tests.rs"]
mod tests;
