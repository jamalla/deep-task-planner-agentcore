#!/usr/bin/env bash
# Build the AgentCore Runtime direct-code-deploy zip (arm64, Python 3.13)
# and upload it to S3. Run in AWS CloudShell from the repo root:
#   bash build.sh                  # bucket: deep-task-planner-agentcore-<account>
set -euo pipefail
cd "$(dirname "$0")"

REGION="${REGION:-us-east-1}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="${BUCKET:-deep-task-planner-agentcore-$ACCOUNT}"
BUILD=/tmp/dtp-agentcore            # /tmp: CloudShell's home is only 1 GB
ZIP=/tmp/dtp-agentcore.zip

command -v uv >/dev/null || pip install -q uv

echo "==> Installing arm64 dependencies (AgentCore Runtime runs on Graviton)"
rm -rf "$BUILD" "$ZIP" && mkdir -p "$BUILD"
UV_CACHE_DIR=/tmp/uv-cache uv pip install -q \
  --python-platform aarch64-manylinux2014 --python-version 3.13 \
  --only-binary=:all: --target "$BUILD" \
  -r requirements.txt

echo "==> Adding our code (main.py must sit at the zip root)"
cp -r app "$BUILD/app"
cp main.py "$BUILD/main.py"
find "$BUILD" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$BUILD" -type d -exec chmod 755 {} +
find "$BUILD" -type f -exec chmod 644 {} +
(cd "$BUILD" && zip -qr "$ZIP" .)
echo "    $(du -h "$ZIP" | cut -f1) zipped, $(du -sh "$BUILD" | cut -f1) unzipped (limits 250 MB / 750 MB)"

echo "==> Uploading to S3"
aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null || \
  aws s3 mb "s3://$BUCKET" --region "$REGION" >/dev/null
aws s3 cp "$ZIP" "s3://$BUCKET/deep-task-planner/deployment_package.zip" --only-show-errors
rm -rf "$BUILD"

echo
echo "Done! In the AgentCore console use this S3 URI:"
echo "    s3://$BUCKET/deep-task-planner/deployment_package.zip"
