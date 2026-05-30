# Launch the CRYCHIC web UI with the model pinned to the 8B Nebius model.
#
# Pins NEBIUS_MODEL to Llama-3.1-8B (overriding whatever is in your session) while
# still inheriting NEBIUS_URL / NEBIUS_API_KEY from your environment, so the Agent
# badge always reads "Nebius · meta-llama/Llama-3.1-8B-Instruct".
#
# Usage:  ./run_web.ps1            # default host 127.0.0.1, port 1200
#         ./run_web.ps1 -Port 8080

param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 1200,
    [string]$Model = "meta-llama/Llama-3.1-8B-Instruct"
)

$env:NEBIUS_MODEL = $Model

if (-not $env:NEBIUS_URL -or -not $env:NEBIUS_API_KEY) {
    Write-Warning "NEBIUS_URL / NEBIUS_API_KEY not set - the agent steps will use the OFFLINE TEMPLATE."
    Write-Warning 'Set them first, e.g.:  $env:NEBIUS_URL="https://api.studio.nebius.com/v1/chat/completions"; $env:NEBIUS_API_KEY="..."'
}

Write-Output ("Starting CRYCHIC web UI on http://{0}:{1}  (model: {2})" -f $BindHost, $Port, $Model)
uvicorn crychic_web:app --host $BindHost --port $Port
