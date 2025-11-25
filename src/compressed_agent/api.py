"""Flask API for compressed agent."""
from flask import Flask, request, jsonify, render_template
from typing import Dict, Any
from .agent import CompressedAgent
from .latent_state import LatentState
from .bedrock_client import BedrockClient
import os

# Template folder is relative to project root
_template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'templates')
app = Flask(__name__, template_folder=_template_dir)

# Initialize Bedrock client (will use fallback if not available)
try:
    bedrock_client = BedrockClient()
except Exception as e:
    print(f"Warning: Could not initialize Bedrock client: {e}. Using fallback mode.")
    bedrock_client = None

agent = CompressedAgent(bedrock_client=bedrock_client)


@app.route("/", methods=["GET"])
def index():
    """Serve the main UI."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route("/api/execute", methods=["POST"])
def execute_goal():
    """Execute a goal."""
    data = request.get_json()
    goal = data.get("goal", "")
    max_steps = data.get("max_steps", 10)
    context = data.get("context", {})

    if not goal:
        return jsonify({"error": "Goal is required"}), 400

    try:
        result = agent.execute_goal(goal, max_steps=max_steps, context=context)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/state", methods=["GET"])
def get_state():
    """Get current state summary."""
    try:
        summary = agent.get_state_summary()
        return jsonify(summary), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/state", methods=["POST"])
def update_state():
    """Update state directly (for testing)."""
    data = request.get_json()
    # This would allow direct state manipulation for testing
    return jsonify({"message": "State update not implemented"}), 501


@app.route("/api/tools", methods=["GET"])
def list_tools():
    """List available tools."""
    try:
        tools = agent.tool_registry.list_tools()
        return jsonify({"tools": tools}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset_agent():
    """Reset agent state."""
    try:
        agent.reset()
        return jsonify({"message": "Agent reset successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)

