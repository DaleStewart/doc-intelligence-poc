import json
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential

from extract import summarize

# Load environment variables from a local .env file (if present) before any
# os.environ lookups below. Real OS env vars take precedence.
load_dotenv()

app = Flask(__name__)

# ---------- Azure AI Document Intelligence ----------
# Set via environment variables (preferred). Key auth is used if DOCINTEL_KEY is set,
# otherwise AAD via DefaultAzureCredential (resource needs 'Cognitive Services User' role).
DOCINTEL_ENDPOINT = os.environ.get("DOCINTEL_ENDPOINT", "")
DOCINTEL_KEY = os.environ.get("DOCINTEL_KEY", "")
DEFAULT_DOCINTEL_MODEL = "prebuilt-layout"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

_credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)


def _docintel_client() -> DocumentIntelligenceClient:
    if not DOCINTEL_ENDPOINT:
        raise RuntimeError("DOCINTEL_ENDPOINT is not set.")
    cred = AzureKeyCredential(DOCINTEL_KEY) if DOCINTEL_KEY else _credential
    return DocumentIntelligenceClient(endpoint=DOCINTEL_ENDPOINT, credential=cred)


def analyze_pdf_bytes(pdf_bytes: bytes, model_id: str = DEFAULT_DOCINTEL_MODEL) -> dict:
    client = _docintel_client()
    # The `keyValuePairs` add-on surfaces label/value pairs that aren't inside
    # tables (e.g. underline-style "Name: ____" fields). It's only supported by
    # prebuilt-layout, so skip it for prebuilt-read and any other model.
    extra: dict = {}
    if model_id == "prebuilt-layout":
        extra["features"] = ["keyValuePairs"]
    poller = client.begin_analyze_document(model_id=model_id, body=pdf_bytes, **extra)
    return poller.result().as_dict()


@app.route("/")
def index():
    return redirect(url_for("document"))


# ---------- Document Intelligence: browser UI ----------
@app.route("/document", methods=["GET", "POST"])
def document():
    error = None
    result_json = None
    summary = None
    model_id = request.values.get("model", DEFAULT_DOCINTEL_MODEL)

    if request.method == "POST":
        file = request.files.get("pdf")
        if not file or file.filename == "":
            error = "Please choose a PDF file."
        else:
            try:
                pdf_bytes = file.read()
                if not pdf_bytes:
                    error = "Uploaded file is empty."
                else:
                    result = analyze_pdf_bytes(pdf_bytes, model_id=model_id)
                    if model_id in ("prebuilt-layout", "prebuilt-read"):
                        summary = summarize(result)
                        result_json = json.dumps(summary, indent=2, default=str)
                    else:
                        # Other models: still show the raw response.
                        result_json = json.dumps(result, indent=2, default=str)
            except HttpResponseError as exc:
                error = f"Document Intelligence error: {exc.message}"
            except Exception as exc:  # noqa: BLE001
                error = f"Failed to analyze PDF: {exc}"

    return render_template(
        "document.html",
        error=error,
        result_json=result_json,
        summary=summary,
        model_id=model_id,
    )


# ---------- Document Intelligence: JSON API ----------
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Analyze a PDF and return the raw Document Intelligence JSON.

    Two ways to call:
      1) multipart/form-data with field 'pdf' (a file)
      2) raw body with Content-Type: application/pdf

    Optional query string: ?model=prebuilt-layout (default)
                           | prebuilt-read | prebuilt-document
                           | prebuilt-invoice | prebuilt-receipt | prebuilt-idDocument | ...
    """
    model_id = request.args.get("model", DEFAULT_DOCINTEL_MODEL)

    if "pdf" in request.files:
        pdf_bytes = request.files["pdf"].read()
    elif request.content_type and "application/pdf" in request.content_type:
        pdf_bytes = request.get_data()
    else:
        return (
            jsonify(
                error="Send a PDF as multipart field 'pdf' or as raw body with Content-Type: application/pdf"
            ),
            400,
        )

    if not pdf_bytes:
        return jsonify(error="Empty PDF body."), 400

    try:
        return jsonify(analyze_pdf_bytes(pdf_bytes, model_id=model_id))
    except HttpResponseError as exc:
        return jsonify(error=exc.message, status=exc.status_code), 502
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=str(exc)), 500


# ---------- Document Intelligence: human-readable summary ----------
@app.route("/api/extract", methods=["POST"])
def api_extract():
    """Run a layout-style model, then return a clean structured summary.

    Query string: ?model=prebuilt-layout (default) | prebuilt-read

    Same input as /api/analyze (multipart 'pdf' or raw application/pdf body).
    """
    model_id = request.args.get("model", "prebuilt-layout")

    if "pdf" in request.files:
        pdf_bytes = request.files["pdf"].read()
    elif request.content_type and "application/pdf" in request.content_type:
        pdf_bytes = request.get_data()
    else:
        return (
            jsonify(
                error="Send a PDF as multipart field 'pdf' or as raw body with Content-Type: application/pdf"
            ),
            400,
        )

    if not pdf_bytes:
        return jsonify(error="Empty PDF body."), 400

    try:
        raw = analyze_pdf_bytes(pdf_bytes, model_id=model_id)
        return jsonify(summarize(raw))
    except HttpResponseError as exc:
        return jsonify(error=exc.message, status=exc.status_code), 502
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=str(exc)), 500


if __name__ == "__main__":
    app.run(debug=True)
