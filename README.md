# Deep Task Planner on Amazon Bedrock AgentCore

The **LangChain deep agent** from [deep-task-planner](https://github.com/jamalla/deep-task-planner) takes a task and breaks it into smart, scheduled action items. This repo hosts it on **Amazon Bedrock AgentCore Runtime**, using the console, instead of AWS Lambda.

```
AgentCore console "Test" / invoke_agent_runtime
        │  {"prompt": "..."} + session id
        ▼
AgentCore Runtime  ── 1 microVM per session (arm64, Python 3.13)
        │  main.py  (BedrockAgentCoreApp, keeps chat history per session)
        ▼
deep agent (app/agent.py) ──▶ OpenAI
   ├─ write_todos
   ├─ task → plan-reviewer sub-agent
   └─ schedule_plan (app/tools.py)
```

## Lambda vs AgentCore Runtime

| | Lambda (deep-task-planner) | AgentCore Runtime (this repo) |
|---|---|---|
| Max run | 15 min per request | 8 h session |
| Isolation | Containers are reused | One microVM per session |
| Chat memory | The browser resends the history | Kept in the session's microVM; send only the new prompt |
| Billing | GB-seconds, including LLM waiting | CPU only while busy; LLM wait costs no CPU (memory is still billed) |
| Auth | Our `x-access-code` | IAM (or JWT) |
| CPU | x86_64 | arm64 |

## Files

| Path | Purpose |
|---|---|
| `main.py` | AgentCore entrypoint: `@app.entrypoint` calls `run_chat` and keeps history per `context.session_id` |
| `app/agent.py`, `app/tools.py` | The same deep agent and tools as the Lambda project |
| `build.sh` | Builds the arm64 direct-code-deploy zip and uploads it to S3 |
| `ui/` | Chat UI: `index.html` + a small Lambda proxy (`handler.py`) that calls `invoke_agent_runtime`, and `deploy_ui.sh` |
| `tests/` | 17 pytest tests (no OpenAI or AWS calls) |

## Lab (console)

1. **Build the zip.** In AWS CloudShell (us-east-1):
   ```bash
   git clone https://github.com/jamalla/deep-task-planner-agentcore.git && cd deep-task-planner-agentcore
   bash build.sh       # prints the S3 URI of deployment_package.zip
   ```
2. **Host it.** Go to the AgentCore console → **Agent Runtime** → **Host agent**, then:
   - Source: S3 → paste the URI
   - Runtime Python 3.13, entry point `main.py`
   - Create a new execution role
   - Environment variables: `OPENAI_API_KEY` (type it yourself) and optionally `OPENAI_MODEL=gpt-5-mini`
3. **Test it** in the console with the same session:
   ```json
   {"prompt": "Plan a 2-day team offsite for 25 people, 6 weeks from now. I can spend about 2 hours a day on weekdays."}
   ```
   then
   ```json
   {"prompt": "Make it weekends only, 5 hours a day"}
   ```
   Send `{"reset": true}` to forget the session's chat.
4. **Chat UI.** In CloudShell, run `bash ui/deploy_ui.sh`. It prints a chat page URL and an access code. The page sends only the new message plus a session id, and AgentCore keeps the history:
   ```
   browser --{prompt, session_id}--> Lambda (ui/handler.py, IAM role: InvokeAgentRuntime on this runtime only)
           --invoke_agent_runtime(runtimeSessionId)--> AgentCore microVM for that session
   ```
5. **Observe it.** The console has a CloudWatch log link per runtime, and the response includes `timings`.
6. **Clean up.** Delete the agent runtime and its endpoint in the console, the `deep-task-planner-agentcore-ui` Lambda and its role, then the S3 bucket and the runtime's execution role.

## Payloads

| Payload | Meaning |
|---|---|
| `{"prompt": "..."}` | New turn in this session (history kept in the microVM) |
| `{"messages": [...]}` | Full history, stateless |
| `{"reset": true}` | Clear this session |

## Run the tests

```bash
pip install -r requirements.txt pytest
python -m pytest -q
```
