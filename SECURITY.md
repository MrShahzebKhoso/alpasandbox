# Security

AlpaSandbox runs a local web app and, optionally, an HTTP model server. Neither is designed to face the internet.

- `alpasandbox app` binds to 127.0.0.1 by default. Only use `--host 0.0.0.0` on a trusted network.
- `alpasandbox serve-model` binds to 0.0.0.0 so another machine can reach it. Set `ALPASANDBOX_TOKEN`, firewall the port, and prefer an SSH tunnel.
- Scenario files are parsed with `yaml.safe_load`. Model plugins are Python code: only install ones you trust.

Please report vulnerabilities privately through GitHub's "Report a vulnerability" (Security tab), not in a public issue.
