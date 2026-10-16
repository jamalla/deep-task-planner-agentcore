#!/usr/bin/env bash
# Deploy the chat UI: a small Lambda + Function URL that serves the page and
# forwards messages to your AgentCore runtime. Run in AWS CloudShell:
#
#   bash ui/deploy_ui.sh                         # finds the runtime named *deep_task_planner*
#   AGENT_RUNTIME_ARN=arn:aws:... bash ui/deploy_ui.sh
#
set -euo pipefail
cd "$(dirname "$0")"

FUNCTION_NAME="${FUNCTION_NAME:-deep-task-planner-agentcore-ui}"
ROLE_NAME="${ROLE_NAME:-deep-task-planner-agentcore-ui-role}"
REGION="${REGION:-us-east-1}"
export AWS_DEFAULT_REGION="$REGION"

# ---------- 1. Which AgentCore runtime? ----------
if [ -z "${AGENT_RUNTIME_ARN:-}" ]; then
  AGENT_RUNTIME_ARN=$(aws bedrock-agentcore-control list-agent-runtimes \
    --query "agentRuntimes[?contains(agentRuntimeName, 'deep_task_planner')] | [0].agentRuntimeArn" --output text)
fi
[ -z "$AGENT_RUNTIME_ARN" ] || [ "$AGENT_RUNTIME_ARN" = "None" ] && { echo "No runtime found; set AGENT_RUNTIME_ARN"; exit 1; }
echo "==> AgentCore runtime: $AGENT_RUNTIME_ARN"

# ---------- 2. Package: handler + page + a recent boto3 (knows bedrock-agentcore) ----------
BUILD=/tmp/dtp-ui-build; ZIP=/tmp/dtp-ui.zip
rm -rf "$BUILD" "$ZIP"
python3 -m pip install --quiet --target "$BUILD" "boto3>=1.40" >/dev/null
cp handler.py index.html "$BUILD/"
find "$BUILD" -name "__pycache__" -type d -prune -exec rm -rf {} +
(cd "$BUILD" && zip -q -r "$ZIP" .)
echo "    ui zip = $(du -h "$ZIP" | cut -f1)"

# ---------- 3. IAM role: logs + invoke ONLY this runtime ----------
if ! ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text 2>/dev/null); then
  echo "==> Creating IAM role $ROLE_NAME"
  ROLE_ARN=$(aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    --query Role.Arn --output text)
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  NEW_ROLE=true
fi
aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name invoke-agentcore-runtime \
  --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"bedrock-agentcore:InvokeAgentRuntime\",\"Resource\":[\"$AGENT_RUNTIME_ARN\",\"$AGENT_RUNTIME_ARN/*\"]}]}"
[ "${NEW_ROLE:-false}" = true ] && { echo "    waiting for IAM to propagate..."; sleep 12; }

# ---------- 4. Create or update the function ----------
EXISTING=$(aws lambda get-function-configuration --function-name "$FUNCTION_NAME" \
  --query 'Environment.Variables' --output json 2>/dev/null || echo null)
python3 - "$EXISTING" "$AGENT_RUNTIME_ARN" > /tmp/dtp-ui-env.json <<'PY'
import json, secrets, sys
env = json.loads(sys.argv[1]) or {}
env.setdefault("ACCESS_CODE", secrets.token_urlsafe(18))
env["AGENT_RUNTIME_ARN"] = sys.argv[2]
print(json.dumps({"Variables": env}))
PY
chmod 600 /tmp/dtp-ui-env.json
if [ "$EXISTING" != "null" ]; then
  echo "==> Updating function"
  aws lambda update-function-code --function-name "$FUNCTION_NAME" --zip-file "fileb://$ZIP" >/dev/null
  aws lambda wait function-updated-v2 --function-name "$FUNCTION_NAME"
  aws lambda update-function-configuration --function-name "$FUNCTION_NAME" \
    --timeout 300 --memory-size 256 --environment file:///tmp/dtp-ui-env.json >/dev/null
  aws lambda wait function-updated-v2 --function-name "$FUNCTION_NAME"
else
  echo "==> Creating function"
  aws lambda create-function --function-name "$FUNCTION_NAME" \
    --runtime python3.13 --handler handler.lambda_handler --role "$ROLE_ARN" \
    --zip-file "fileb://$ZIP" --timeout 300 --memory-size 256 \
    --environment file:///tmp/dtp-ui-env.json >/dev/null
  aws lambda wait function-active-v2 --function-name "$FUNCTION_NAME"
fi
rm -f /tmp/dtp-ui-env.json; rm -rf "$BUILD"

# ---------- 5. Public URL (the API itself needs the access code) ----------
if ! URL=$(aws lambda get-function-url-config --function-name "$FUNCTION_NAME" --query FunctionUrl --output text 2>/dev/null); then
  URL=$(aws lambda create-function-url-config --function-name "$FUNCTION_NAME" \
    --auth-type NONE --query FunctionUrl --output text)
  aws lambda add-permission --function-name "$FUNCTION_NAME" --statement-id PublicUrl \
    --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$FUNCTION_NAME" --statement-id PublicUrlInvoke \
    --action lambda:InvokeFunction --principal "*" --invoked-via-function-url >/dev/null 2>&1 || true
fi
curl -s -o /dev/null -w "    GET /health -> HTTP %{http_code}\n" "${URL}health" || true

CODE=$(aws lambda get-function-configuration --function-name "$FUNCTION_NAME" \
  --query Environment.Variables.ACCESS_CODE --output text)
echo
echo "==> Done!"
echo "    Chat page:   $URL"
echo "    Access code: $CODE   (paste it in the chat page, top right)"
