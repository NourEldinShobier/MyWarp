//! Compaction sidecar state - attached to `AIConversation`, decoupled from the warp `api::Message` protocol.
//!
//! Because warp's `api::Message` comes from an external protobuf dependency (`warp_multi_agent_api`),
//! fields such as `is_summary` / `compacted` cannot be added to it; this sidecar is indexed by message_id
//! and holds this "compaction metadata" on the conversation side.
//!
//! The serialization version [`CompactionState::VERSION`] is bumped manually when the schema evolves;
//! old conversations that fail to deserialize fall back to `Default` (equivalent to "never compacted").

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

/// Source that triggered compaction. `Auto` is triggered only by token overflow; `Manual` is /compact /compact-and.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CompactionTrigger {
    Manual,
    Auto,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MessageMarker {
    /// This assistant message is a summary; its content replaces the earlier history when the request is assembled.
    #[serde(default)]
    pub is_summary: bool,
    /// This user message is a compaction trigger placeholder (opencode `parts.some(p => p.type === "compaction")`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction_trigger: Option<CompactionTrigger>,
    /// This ToolCallResult's output has been pruned; it is replaced with a placeholder when projected. Unix epoch ms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_output_compacted_at: Option<u64>,
    /// Marker for the synthetic "Continue..." user message created on auto-continue
    /// (matches opencode `metadata.compaction_continue`).
    #[serde(default)]
    pub synthetic_continue: bool,
}

/// A completed compaction span (matches the items returned by opencode `completedCompactions()`).
///
/// `user_msg_id` is the user message that triggered the summary (carrying the compaction_trigger marker),
/// `assistant_msg_id` is the synthesized summary AgentOutput message. Both are treated as covered in [`CompactionState::hidden_message_ids`]
/// and skipped when projecting - but the summary text itself is pulled out and substituted into the head area.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedCompaction {
    pub user_msg_id: String,
    pub assistant_msg_id: String,
    /// Head-area message ids covered by this summary; all hidden when projecting a normal request.
    #[serde(default)]
    pub head_message_ids: Vec<String>,
    /// Message id of the tail start, used for split verification / debugging.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tail_start_id: Option<String>,
    /// Summary content (could be read directly from the assistant message, but is cached in state so build_prompt can get previous_summary).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary_text: Option<String>,
    pub auto: bool,
    pub overflow: bool,
}

/// Sidecar table persisted together with `AIConversation`.
///
/// Default value = empty table = uncompacted state, completely non-invasive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionState {
    /// Schema version, bumped when it evolves.
    #[serde(default = "CompactionState::current_version")]
    pub version: u32,
    #[serde(default)]
    markers: HashMap<String, MessageMarker>,
    #[serde(default)]
    completed: Vec<CompletedCompaction>,
}

impl Default for CompactionState {
    fn default() -> Self {
        Self {
            version: Self::VERSION,
            markers: HashMap::new(),
            completed: Vec::new(),
        }
    }
}

impl CompactionState {
    pub const VERSION: u32 = 2;
    fn current_version() -> u32 {
        Self::VERSION
    }

    pub fn marker(&self, msg_id: &str) -> Option<&MessageMarker> {
        self.markers.get(msg_id)
    }

    /// Writes a marker (merges into an existing marker instead of overwriting the whole marker).
    pub fn upsert_marker(&mut self, msg_id: impl Into<String>, f: impl FnOnce(&mut MessageMarker)) {
        let entry = self.markers.entry(msg_id.into()).or_default();
        f(entry);
    }

    /// Marks that a ToolCallResult's output has been pruned.
    pub fn mark_tool_compacted(&mut self, msg_id: impl Into<String>, now_ms: u64) {
        self.upsert_marker(msg_id, |m| m.tool_output_compacted_at = Some(now_ms));
    }

    /// Pushes one completed compaction.
    pub fn push_completed(&mut self, c: CompletedCompaction) {
        // Also set a marker on the user and assistant messages separately (so they can be identified individually when projecting).
        self.upsert_marker(c.user_msg_id.clone(), |m| {
            m.compaction_trigger = Some(if c.auto {
                CompactionTrigger::Auto
            } else {
                CompactionTrigger::Manual
            });
        });
        self.upsert_marker(c.assistant_msg_id.clone(), |m| m.is_summary = true);
        self.completed.push(c);
    }

    /// Marks a synthetic "Continue..." user message (created on the auto+overflow path).
    pub fn mark_synthetic_continue(&mut self, msg_id: impl Into<String>) {
        self.upsert_marker(msg_id, |m| m.synthetic_continue = true);
    }

    /// Gets the last completed compaction (the incremental-summary anchor for [`super::prompt::build_prompt`]).
    pub fn previous_summary(&self) -> Option<&str> {
        self.completed
            .last()
            .and_then(|c| c.summary_text.as_deref())
    }

    pub fn completed(&self) -> &[CompletedCompaction] {
        &self.completed
    }

    /// All message ids to skip when assembling a request (matches opencode `hidden`):
    /// head_message_ids + user_msg_id + assistant_msg_id of every span of completed compaction.
    ///
    /// Note: this is only the "set of message ids originally to be hidden from history" and does **not** include the summary itself -
    /// the summary text is inserted by the request projection as a synthesized message at the compaction trigger user_msg_id position.
    pub fn hidden_message_ids(&self) -> HashSet<String> {
        let mut out = HashSet::new();
        for c in &self.completed {
            out.extend(c.head_message_ids.iter().cloned());
            out.insert(c.user_msg_id.clone());
            out.insert(c.assistant_msg_id.clone());
        }
        out
    }

    /// Debug / test entry: checks whether a marker exists.
    #[cfg(test)]
    pub(crate) fn marker_count(&self) -> usize {
        self.markers.len()
    }
}

#[cfg(test)]
mod state_tests {
    use super::*;

    fn cc(uid: &str, aid: &str, auto: bool) -> CompletedCompaction {
        CompletedCompaction {
            user_msg_id: uid.to_string(),
            assistant_msg_id: aid.to_string(),
            head_message_ids: Vec::new(),
            tail_start_id: None,
            summary_text: Some(format!("summary-{aid}")),
            auto,
            overflow: false,
        }
    }

    #[test]
    fn push_completed_marks_both_messages() {
        let mut s = CompactionState::default();
        s.push_completed(cc("u1", "a1", true));
        assert!(s.marker("u1").unwrap().compaction_trigger == Some(CompactionTrigger::Auto));
        assert!(s.marker("a1").unwrap().is_summary);
    }

    #[test]
    fn previous_summary_returns_last() {
        let mut s = CompactionState::default();
        s.push_completed(cc("u1", "a1", false));
        s.push_completed(cc("u2", "a2", false));
        assert_eq!(s.previous_summary(), Some("summary-a2"));
    }

    #[test]
    fn hidden_message_ids_covers_all_completed() {
        let mut s = CompactionState::default();
        s.push_completed(cc("u1", "a1", false));
        s.push_completed(cc("u2", "a2", false));
        let h = s.hidden_message_ids();
        assert!(h.contains("u1"));
        assert!(h.contains("a1"));
        assert!(h.contains("u2"));
        assert!(h.contains("a2"));
        assert_eq!(h.len(), 4);
    }

    #[test]
    fn hidden_message_ids_includes_head_message_ids() {
        let mut s = CompactionState::default();
        let mut c = cc("u1", "a1", false);
        c.head_message_ids = vec!["h1".to_string(), "h2".to_string(), "u1".to_string()];
        s.push_completed(c);
        let h = s.hidden_message_ids();
        assert!(h.contains("h1"));
        assert!(h.contains("h2"));
        assert!(h.contains("u1"));
        assert!(h.contains("a1"));
        assert_eq!(h.len(), 4);
    }

    #[test]
    fn v1_completed_compaction_deserializes_to_empty_head_message_ids() {
        let json = r#"{
            "user_msg_id":"u1",
            "assistant_msg_id":"a1",
            "tail_start_id":null,
            "summary_text":"summary",
            "auto":false,
            "overflow":false
        }"#;
        let c: CompletedCompaction = serde_json::from_str(json).unwrap();
        assert!(c.head_message_ids.is_empty());
    }

    #[test]
    fn upsert_marker_merges() {
        let mut s = CompactionState::default();
        s.upsert_marker("m1", |m| m.is_summary = true);
        s.upsert_marker("m1", |m| m.synthetic_continue = true);
        let m = s.marker("m1").unwrap();
        assert!(m.is_summary);
        assert!(m.synthetic_continue);
        assert_eq!(s.marker_count(), 1);
    }

    #[test]
    fn default_serializable_roundtrip() {
        let s = CompactionState::default();
        let j = serde_json::to_string(&s).unwrap();
        let back: CompactionState = serde_json::from_str(&j).unwrap();
        assert_eq!(back.version, CompactionState::VERSION);
        assert!(back.completed.is_empty());
    }
}
