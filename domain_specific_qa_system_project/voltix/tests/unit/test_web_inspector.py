from unittest.mock import patch, MagicMock
from app.tools.web_inspector import fetch_local_web_page

def test_web_inspector_success():
    mock_html = """
    <html>
        <head><title>Test EEE Documentation</title></head>
        <body>
            <h1>Transformer Efficiency</h1>
            <p>The maximum efficiency of a transformer occurs when copper loss equals iron loss.</p>
            <a href="/docs/induction">Induction Motors</a>
        </body>
    </html>
    """
    mock_res = MagicMock()
    mock_res.text = mock_html
    mock_res.status_code = 200
    mock_res.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_res):
        result = fetch_local_web_page("http://127.0.0.1:5000/docs")
        assert result["status"] == "success"
        assert result["title"] == "Test EEE Documentation"
        assert "Transformer Efficiency" in result["headings"]
        assert "maximum efficiency" in result["content"]

def test_web_inspector_network_error():
    with patch("requests.get", side_effect=Exception("Connection refused")):
        result = fetch_local_web_page("http://invalid.local.url:9999")
        assert result["status"] == "error"
        assert "Failed to fetch web page" in result["error"]
