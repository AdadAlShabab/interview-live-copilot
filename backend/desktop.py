"""Launch the FastAPI application in a native PyWebView window."""

import threading

import uvicorn


def main() -> None:
    import webview

    from app.config.settings import get_settings

    settings = get_settings()
    server = uvicorn.Server(
        uvicorn.Config(
            "app.main:app",
            host=settings.backend_host,
            port=settings.backend_port,
            log_level=settings.log_level.lower(),
        )
    )
    threading.Thread(target=server.run, daemon=True).start()
    webview.create_window(
        "Interview Copilot",
        settings.base_url + "/ui/",
        width=1280,
        height=820,
        frameless=True,
        easy_drag=True,
        shadow=False,
        on_top=True,
    )
    webview.start()


if __name__ == "__main__":
    main()