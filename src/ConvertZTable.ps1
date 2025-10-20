$inputFile = "zTable.txt"
$outputFile = "z_table.lua"

$lines = Get-Content $inputFile
$headers = $lines[0] -split "`t"

Add-Content -Path $outputFile -Value "local zTableDictionary = {"

for ($i=1; $i -lt $lines.Count; $i++) {
    $cols = $lines[$i] -split "`t"
    $rowKey = $cols[0]
    Add-Content -Path $outputFile -Value "  [""$rowKey""] = {"
    for ($j=1; $j -lt $cols.Count; $j++) {
        $header = $headers[$j]
        $value = $cols[$j]
        Add-Content -Path $outputFile -Value "    [""$header""] = $value,"
    }
    Add-Content -Path $outputFile -Value "  },"
}

Add-Content -Path $outputFile -Value "}"

Write-Host "Conversion done! Output written to $outputFile"