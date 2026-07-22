$ErrorActionPreference = "Stop"

$Conda = "C:\Users\Asus\anaconda3\Scripts\conda.exe"

Write-Host "1) Select access point from pivot table with conda base..."
& $Conda run -n base python .\01_select_access_point.py

Write-Host "2) Draw selected AP RSS values on the JSON faculty map with conda base..."
& $Conda run -n base python .\02_plot_selected_ap_from_pivot.py

Write-Host "3) Fit selected AP location with conda gaussianfit..."
& $Conda run -n gaussianfit python .\03_fit_selected_ap_location.py

Write-Host "Done. Outputs are in .\outputs_rss_ap"
