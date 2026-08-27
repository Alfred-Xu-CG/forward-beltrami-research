param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs
)

$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:PYTHONPATH = Join-Path $PSScriptRoot '..\src'
$python = 'C:\Users\xuzhehao\anaconda3\python.exe'
& $python @PythonArgs
exit $LASTEXITCODE
