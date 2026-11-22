//! This module contains all code relevant to Agent Predict within Warp.
//!
//! Agent Predict attempts to predict the next action the user will take in Warp.

pub(crate) mod generate_ai_input_suggestions;
pub(crate) mod generate_am_query_suggestions;
pub mod next_command_model;
// Warp (Wave 3-2): the `predict_am_queries` API module has been physically deleted. The original
// `ServerApi::predict_am_queries` had 0 external consumers and was removed with it;
// FeatureFlag::PredictAMQueries and `predict_am_queries_future_handle` in terminal/input.rs
// are kept only as a control switch / handle name and no longer need this module.
pub mod prompt_suggestions;
