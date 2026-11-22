use super::*;

#[test]
fn tags_become_chips() {
    assert_eq!(style_line("[[Done]] shipped"), "`Done` shipped");
    assert_eq!(style_line("[[Passed 12/12]]"), "`Passed 12/12`");
    assert_eq!(style_line("[[next: run it]]"), "`next: run it`");
}

#[test]
fn inline_code_and_non_tags_are_untouched() {
    assert_eq!(style_line("use `[[Done]]` here"), "use `[[Done]]` here");
    assert_eq!(style_line("arr[[1]] and [[ ]]"), "arr[[1]] and [[ ]]");
    assert_eq!(style_line("[[unclosed"), "[[unclosed");
}

#[test]
fn fences_are_skipped_and_chunks_are_buffered() {
    let mut s = LineStyler::default();
    assert_eq!(s.push("[[Do"), "");
    assert_eq!(s.push("ne]] ok\n```\n[[Done]]\n"), "`Done` ok\n```\n[[Done]]\n");
    assert_eq!(s.push("```\n[[Failed]]"), "```\n");
    assert_eq!(s.flush(), "`Failed`");
}

#[test]
fn lead_ins_are_bolded() {
    assert_eq!(bold_lead_in("Next: run it"), "**Next:** run it");
    assert_eq!(bold_lead_in("- Risk: data loss"), "- **Risk:** data loss");
    assert_eq!(bold_lead_in("  fix: use a lock\n"), "  **fix:** use a lock\n");
    assert_eq!(bold_lead_in("Nextly: nope"), "Nextly: nope");
    assert_eq!(bold_lead_in("see Next: later"), "see Next: later");
}
