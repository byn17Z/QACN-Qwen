"""
Description: Gradio web UI for model deployment.
Provides a chat interface with RAG toggle and generation parameter controls.
Usage: from src.gradio_app import create_gradio_app
Dependencies: gradio
"""

from typing import Any, Dict

import gradio as gr

from src.inference_engine import InferenceEngine


def create_gradio_app(engine: InferenceEngine, deploy_config: Dict[str, Any]) -> gr.Blocks:
    """
    Create Gradio chat interface with RAG toggle.

    Args:
        engine: The InferenceEngine instance.
        deploy_config: The deploy section of config.

    Returns:
        gr.Blocks instance ready to mount on FastAPI.
    """
    rag_default = deploy_config.get("rag_enabled_default", False)
    default_max_tokens = int(deploy_config.get("max_new_tokens", 512))
    default_temperature = float(deploy_config.get("temperature", 0.7))

    def respond(
        message: str,
        chat_history: list,
        use_rag: bool,
        max_tokens: int,
        temperature: float,
    ) -> tuple[list, str]:
        """Process user message and return updated chat history."""
        if not message.strip():
            return chat_history, ""

        result = engine.generate(
            instruction=message,
            use_rag=use_rag,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        response = result["response"]

        # Append sources if RAG was used
        if result["context_used"]:
            sources = "\n\n---\n**Sources:**\n"
            for i, doc in enumerate(result["context_used"], 1):
                # Truncate long documents for display
                truncated = doc[:200] + "..." if len(doc) > 200 else doc
                sources += f"{i}. {truncated}\n"
            response += sources

        chat_history.append((message, response))
        return chat_history, ""

    with gr.Blocks(title="QEdpediaCN-Qwen") as demo:
        gr.Markdown("# QEdpediaCN-Qwen Educational QA System")

        chatbot = gr.Chatbot(label="Chat", height=400)

        with gr.Row():
            msg = gr.Textbox(
                label="Question",
                placeholder="Enter your question here...",
                scale=4,
            )
            submit_btn = gr.Button("Submit", variant="primary", scale=1)

        with gr.Row():
            use_rag = gr.Checkbox(label="Enable RAG", value=rag_default)
            max_tokens = gr.Slider(
                minimum=64, maximum=2048, step=64,
                value=default_max_tokens, label="Max Tokens",
            )
            temperature = gr.Slider(
                minimum=0.0, maximum=2.0, step=0.1,
                value=default_temperature, label="Temperature",
            )

        clear_btn = gr.Button("Clear")

        # Wire up events
        submit_btn.click(
            respond,
            inputs=[msg, chatbot, use_rag, max_tokens, temperature],
            outputs=[chatbot, msg],
        )
        msg.submit(
            respond,
            inputs=[msg, chatbot, use_rag, max_tokens, temperature],
            outputs=[chatbot, msg],
        )
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

    return demo
