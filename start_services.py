
#!/usr/bin/env python3
"""
start_services.py

Starts the production AI stack using a single docker-compose.yml.
Supabase is remote, so no local Supabase services are started.
"""

import subprocess

PROJECT_NAME = "localai"
COMPOSE_FILE = "docker-compose.yml"


def run_command(cmd):
    """Run a shell command and print it."""
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def stop_existing_containers():
    """Stop and remove existing containers and orphans."""
    print(f"Stopping existing containers for project '{PROJECT_NAME}'...")
    run_command([
        "docker", "compose",
        "-p", PROJECT_NAME,
        "-f", COMPOSE_FILE,
        "down",
        "--remove-orphans"
    ])


def start_services():
    """Start services using the main docker-compose file."""
    print("Starting services...")
    run_command([
        "docker", "compose",
        "-p", PROJECT_NAME,
        "-f", COMPOSE_FILE,
        "up",
        "-d"
    ])


def main():
    stop_existing_containers()
    start_services()


if __name__ == "__main__":
    main()

