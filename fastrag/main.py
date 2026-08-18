import argparse
import logging
import sys

import uvicorn
import yaml

from .runtime_parameters import RuntimeParametersLoader
from .settings import load_settings


def parse_args() -> argparse.Namespace:
    defaults = load_settings()
    parser = argparse.ArgumentParser(description="FastAPI Custom Model Runner (Fast DRUM)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    server_parser = subparsers.add_parser("server", help="Start the prediction server")

    server_parser.add_argument(
        "-cd",
        "--code-dir",
        default=defaults.code_dir,
        help="Directory containing custom model code (custom.py)",
    )
    server_parser.add_argument(
        "--address", default=defaults.address, help="Address to bind to (host:port)"
    )
    server_parser.add_argument(
        "--max-workers",
        dest="workers",
        type=int,
        default=defaults.workers,
        help="Number of worker processes",
    )
    server_parser.add_argument("--runtime-params-file", help="Path to runtime parameters YAML file")
    server_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    server_parser.add_argument(
        "--allow-dr-api-access", action="store_true", help="Allow DataRobot API access"
    )

    return parser.parse_args()


def run_server(args: argparse.Namespace) -> None:
    settings = load_settings()
    # Update settings from CLI args
    for key, value in vars(args).items():
        if hasattr(settings, key) and value is not None:
            setattr(settings, key, value)

    # Without an explicit level the root logger stays at WARNING, which silently hides
    # every logger.info in the package -- including whether stats reporting is working.
    logging.basicConfig(level=logging.DEBUG if settings.verbose else logging.INFO)
    logger = logging.getLogger("fastrag.main")

    # Set up environment variables via Settings helper, so any overirdes are propagated
    # to uvicorn workers.
    settings.setup_env()

    logger.info(f"Starting Fast DRUM with code_dir: {settings.code_dir}")

    if settings.runtime_params_file:
        logger.info(f"Loading runtime parameters from {settings.runtime_params_file}")
        try:
            loader = RuntimeParametersLoader(settings.runtime_params_file, settings.code_dir)
            loader.setup_environment_variables()
            logger.info("Runtime parameters loaded successfully.")
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"Failed to load runtime parameters: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error loading runtime parameters: {e}")
            sys.exit(1)

    try:
        host, str_port = settings.address.split(":")
        port = int(str_port)
    except ValueError:
        logger.error("Invalid address format. Use host:port")
        sys.exit(1)

    logger.info(f"Running server on {host}:{port} with {settings.workers} workers")

    uvicorn.run(
        "fastrag.server:app",
        host=host,
        port=port,
        workers=settings.workers,
        log_level="debug" if settings.verbose else "info",
        forwarded_allow_ips="*",
    )


def main() -> None:
    args = parse_args()
    if args.command == "server":
        run_server(args)


if __name__ == "__main__":
    main()
