# GroundingDINO Nuclio Function

This folder contains a Nuclio serverless function for **GroundingDINO** zero-shot object detection.

## Files
- `main.py` — Python source code for the function
- `function.yaml` — Nuclio function configuration

## Development
Edit `main.py` in your IDE. You can add dependencies to the `build.commands` section in `function.yaml` if needed.

## Deployment
To deploy with `nuctl`:
```bash
nuctl deploy --path . --project-name cvat --namespace nuclio
```

## Testing
Once deployed, send a request:
```bash
curl -X POST -H "Content-Type: application/json" \
    -d '{"image_url": "https://example.com/test.jpg", "prompt": "a person"}' \
    http://<nuclio-gateway-ip>:<port>/
```

Replace `<nuclio-gateway-ip>:<port>` with your Nuclio endpoint.


## Install nuctl

```bash
curl -s https://api.github.com/repos/nuclio/nuclio/releases/latest \
			| grep -i "browser_download_url.*nuctl.*$(uname)" \
			| cut -d : -f 2,3 \
			| tr -d \" \
			| wget -O nuctl -qi - && chmod +x nuctl

sudo mv nuctl /usr/local/bin/
```
