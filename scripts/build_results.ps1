param([string]$Root = (Split-Path $PSScriptRoot -Parent))
$ErrorActionPreference = 'Stop'
$culture = [Globalization.CultureInfo]::InvariantCulture
$summaryDir = Join-Path $Root 'reports/summary'
$figureDir = Join-Path $Root 'reports/figures'
New-Item -ItemType Directory -Path $summaryDir,$figureDir -Force | Out-Null
function Read-Json([string]$path) {
    Get-Content -LiteralPath (Join-Path $Root $path) -Raw -Encoding UTF8 | ConvertFrom-Json
}
function Export-Rows($rows, [string]$name) {
    $rows | Export-Csv -LiteralPath (Join-Path $summaryDir $name) -NoTypeInformation -Encoding UTF8
}
function Percent($value) { ([double]$value * 100).ToString('F2',$culture) + '%' }
function Write-Utf8([string]$path,[string]$text) {
    [IO.File]::WriteAllText($path,$text,[Text.UTF8Encoding]::new($false))
}
$asrMap = [ordered]@{
    'openai/whisper-tiny' = @('A0','Whisper-tiny สำเร็จรูป')
    'models/whisper-th' = @('A1','Whisper-tiny ปรับด้วยภาษาไทย')
    'openai/whisper-small' = @('A2','Whisper-small สำเร็จรูป')
    'models/whisper-small-th' = @('A3','Whisper-small + LoRA + เสียงจาก gTTS')
    'models/whisper-small-th-v2' = @('A4','Whisper-small + เสียงผู้ใช้ 72 ประโยค')
    'models/whisper-small-th-v3' = @('A5','Whisper-small + เสียงผู้ใช้ 112 ประโยค')
    'models/whisper-small-th-v4' = @('A6','Whisper-small + เสียงผู้ใช้ 112 ประโยค + ฝึกต่อ')
}
$intentMap = [ordered]@{
    nb='Naive Bayes'; logreg='Logistic Regression (คำ)'
    logreg_char='Logistic Regression (คำ + ตัวอักษร)'; cnn='TextCNN'; lstm='BiLSTM'
}
$asrIndex = @{}
foreach($file in Get-ChildItem (Join-Path $Root 'reports/asr') -Filter '*__fleurs.json') {
    $relative = 'reports/asr/' + $file.Name
    $data = Read-Json $relative
    $model = $data.model
    if(!$asrMap.Contains($model)) { throw "Unknown ASR model: $model" }
    $asrIndex["FLEURS|$model"] = [pscustomobject][ordered]@{
        model_id=$asrMap[$model][0]; model=$asrMap[$model][1]; model_path=$model
        test_set='FLEURS'; n=$data.n; cer=$data.cer; wer=$data.wer
        exact_match=$null; exact_correct=$null; source_file=$relative
    }
}
# Later snapshots take precedence for duplicate ASR measurements on the same test set.
$runs = @(
    @('summary_v1.json','I1'), @('summary_v2.json','I2'), @('summary_v3_24.json','I3'),
    @('summary_asrv2_60.json','I3'), @('summary_asrv3_60.json','I4'), @('summary_asrv4_60.json','I4')
)
$voice = @()
foreach($run in $runs) {
    $relative = 'reports/e2e/' + $run[0]
    $data = Read-Json $relative
    foreach($prop in $data.asr.PSObject.Properties) {
        $model=$prop.Name; $value=$prop.Value
        $asrIndex["voice_$($data.n)|$model"] = [pscustomobject][ordered]@{
            model_id=$asrMap[$model][0]; model=$asrMap[$model][1]; model_path=$model
            test_set="voice_$($data.n)"; n=$data.n; cer=$value.cer; wer=$value.wer
            exact_match=$value.exact_match; exact_correct=[int][math]::Round($value.exact_match*$data.n)
            source_file=$relative
        }
    }
    foreach($classifier in $data.intent_accuracy.PSObject.Properties) {
        foreach($prop in $classifier.Value.PSObject.Properties) {
            $id = if($prop.Name -eq 'reference') {'reference'} else {$asrMap[$prop.Name][0]}
            $voice += [pscustomobject][ordered]@{
                model_key=$classifier.Name; model=$intentMap[$classifier.Name]; training_set=$run[1]
                run=$run[0]; text_source=$id; test_set="voice_$($data.n)"; n=$data.n
                accuracy=$prop.Value; correct=[int][math]::Round($prop.Value*$data.n); source_file=$relative
            }
        }
    }
}
$asr = @($asrIndex.Values | Sort-Object test_set,model_id)
$massive = @()
foreach($run in @(@('metrics_v1.json','I1'),@('metrics_v2.json','I2'),@('metrics_v3.json','I3'),@('metrics.json','I4'))) {
    $relative='reports/intent/'+$run[0]; $data=Read-Json $relative
    foreach($key in $intentMap.Keys) {
        $value=$data.$key
        if($null -eq $value) { continue }
        $massive += [pscustomobject][ordered]@{
            model_key=$key; model=$intentMap[$key]; training_set=$run[1]; test_set='MASSIVE th-TH + en-US'
            accuracy=$value.accuracy; macro_f1=$value.macro_f1
            accuracy_th=$value.'accuracy_th-TH'; accuracy_en=$value.'accuracy_en-US'; source_file=$relative
        }
    }
}
$latest=Read-Json 'reports/e2e/summary_asrv4_60.json'
$utterances=@(Import-Csv -LiteralPath (Join-Path $Root 'reports/e2e/utterances_asrv4_60.csv') -Encoding UTF8)
# Validate each classifier's summary against the saved per-command predictions.
foreach($key in $intentMap.Keys) {
    foreach($prop in $latest.intent_accuracy.$key.PSObject.Properties) {
        $source=Split-Path $prop.Name -Leaf
        $rows=@($utterances | Where-Object source -eq $source)
        $correct=@($rows | Where-Object { $_."pred_$key" -eq $_.gold_intent }).Count
        if($rows.Count -ne $latest.n -or [math]::Abs($correct/$rows.Count-$prop.Value) -gt 1e-10) {
            throw "Prediction/summary mismatch: $key $source"
        }
    }
}
$system=@(); $perCommand=@()
foreach($version in 2,3,4) {
    $relative="reports/e2e/slots_asrv$version.json"; $slots=Read-Json $relative
    $source=$slots.asr; $id=$asrMap["models/$source"][0]
    $details=@($slots.detail | Where-Object source -eq $source)
    $rows=@($utterances | Where-Object source -eq $source)
    $reads=@($rows | Where-Object gold_intent -notin @('add_task','complete_task','add_event'))
    $readCorrect=@($reads | Where-Object {$_.pred_nb -eq $_.gold_intent}).Count
    $writeCorrect=@($details | Where-Object ok_success).Count
    $slotsCorrect=@($details | Where-Object ok_all_slots).Count
    foreach($detail in $details) {
        $row=@($rows | Where-Object file -eq $detail.file)
        if($row.Count -ne 1 -or $row[0].text -ne $detail.text -or $row[0].pred_nb -ne $detail.pred_intent) {
            throw "Slot/prediction mismatch: $source $($detail.file)"
        }
    }
    if($rows.Count -ne 60 -or $details.Count -ne 30 -or $reads.Count -ne 30) { throw 'Unexpected sample counts' }
    $system += [pscustomobject][ordered]@{
        asr_id=$id; intent_model='Naive Bayes'; training_set='I4'; n=60
        intent_correct=@($rows | Where-Object {$_.pred_nb -eq $_.gold_intent}).Count
        intent_accuracy=$latest.intent_accuracy.nb."models/$source"
        slot_n=30; all_slots_correct=$slotsCorrect; all_slots_accuracy=$slotsCorrect/30
        write_success_correct=$writeCorrect; write_success_accuracy=$writeCorrect/30
        read_correct=$readCorrect; read_n=30; system_correct=$readCorrect+$writeCorrect
        system_accuracy=($readCorrect+$writeCorrect)/60
        source_file=$relative; prediction_file='reports/e2e/utterances_asrv4_60.csv'
    }
    foreach($group in 'add_task','complete_task','add_event') {
        $v=$slots.results.$source.$group
        $perCommand += [pscustomobject][ordered]@{
            asr_id=$id; command=$group; n=$v.n; success_correct=[int][math]::Round($v.success*$v.n)
            success_accuracy=$v.success; all_slots_accuracy=$v.all_slots; source_file=$relative
        }
    }
}
Export-Rows $asr 'asr_results.csv'
Export-Rows $massive 'intent_massive_results.csv'
Export-Rows $voice 'intent_voice_results.csv'
Export-Rows $system 'system_results.csv'
Export-Rows $perCommand 'system_per_command.csv'
Write-Utf8 (Join-Path $summaryDir 'results.json') (([ordered]@{
    asr=$asr; intent_massive=$massive; intent_voice=$voice; system=$system; system_per_command=$perCommand
    notes=@(
        'Metrics are read from saved experiments; no models are retrained.',
        'I1-I4 labels follow the experiment mapping in report.md; original JSON files do not store training-set IDs.',
        'Accuracy and exact_match: higher is better. CER and WER: lower is better. CER/WER are not accuracy.',
        'System correctness joins 30 read/other classifications with 30 write-command intent+slot successes.',
        'Slot tests use simulated open tasks. System correctness does not measure successful Google API writes.'
    )
}) | ConvertTo-Json -Depth 10)

# Render charts using the built-in .NET plotting library (no GUI windows).
# Design: one emphasis hue (blue) against neutral gray, horizontal rows with direct labels,
# takeaway titles, and shared scales across figures that measure the same thing.
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing
$fontName='Leelawadee UI'
$palette=@{
    ink='#0b0b0b'; ink2='#52514e'; muted='#898781'; grid='#e1e0d9'; axis='#c3c2b7'
    accent='#2a78d6'; accentLight='#86b6ef'; accentDark='#1c5cab'; context='#cfcec8'; second='#eb6834'
}
function Color([string]$name) { [Drawing.ColorTranslator]::FromHtml($palette[$name]) }
function Font([double]$px,[switch]$Bold) {
    $style=if($Bold){[Drawing.FontStyle]::Bold}else{[Drawing.FontStyle]::Regular}
    [Drawing.Font]::new($fontName,[single]$px,$style,[Drawing.GraphicsUnit]::Pixel)
}
function Add-Text($chart,[string]$text,[double]$x,[double]$y,[double]$w,[double]$h,$font,[string]$color,[string]$align='MiddleLeft') {
    $a=[System.Windows.Forms.DataVisualization.Charting.TextAnnotation]::new()
    $a.Text=$text; $a.Font=$font; $a.ForeColor=Color $color; $a.Alignment=$align
    $a.X=100*$x/$chart.Width; $a.Y=100*$y/$chart.Height; $a.Width=100*$w/$chart.Width; $a.Height=100*$h/$chart.Height
    $chart.Annotations.Add($a)
}
function New-Canvas([int]$width,[int]$height,[string]$title,[string]$subtitle) {
    $chart=[System.Windows.Forms.DataVisualization.Charting.Chart]::new()
    $chart.Width=$width; $chart.Height=$height; $chart.BackColor=[Drawing.Color]::White
    $chart.AntiAliasing='All'; $chart.TextAntiAliasingQuality='High'
    Add-Text $chart $title 24 16 ($width-48) 54 (Font 38 -Bold) 'ink'
    Add-Text $chart $subtitle 24 70 ($width-48) 40 (Font 27) 'ink2'
    $chart
}
function Add-Area($chart,[string]$name,[double]$x,[double]$y,[double]$w,[double]$h,[double]$left,[double]$bottom) {
    $area=[System.Windows.Forms.DataVisualization.Charting.ChartArea]::new($name)
    $area.BackColor=[Drawing.Color]::Transparent
    $area.Position.Auto=$false
    $area.Position.X=100*$x/$chart.Width; $area.Position.Y=100*$y/$chart.Height
    $area.Position.Width=100*$w/$chart.Width; $area.Position.Height=100*$h/$chart.Height
    $area.InnerPlotPosition.Auto=$false
    $area.InnerPlotPosition.X=100*$left/$w; $area.InnerPlotPosition.Y=0
    $area.InnerPlotPosition.Width=100*($w-$left)/$w; $area.InnerPlotPosition.Height=100*($h-$bottom)/$h
    foreach($axis in $area.AxisX,$area.AxisY) {
        $axis.MajorGrid.Enabled=$false; $axis.MajorTickMark.Enabled=$false; $axis.IsLabelAutoFit=$false
        $axis.LineColor=Color 'axis'; $axis.LabelStyle.Font=Font 25; $axis.LabelStyle.ForeColor=Color 'muted'
        $axis.IsMarginVisible=$false
    }
    $chart.ChartAreas.Add($area)
    $area
}
# Value axis with gridlines and tick labels only at $ticks (the axis may extend past them to fit labels).
function Set-ValueAxis($axis,[double]$min,[double]$max,[double[]]$ticks) {
    $axis.Minimum=$min; $axis.Maximum=$max; $axis.LineColor=[Drawing.Color]::Transparent
    $span=0.48*(($ticks[1..($ticks.Count-1)] | ForEach-Object -Begin {$prev=$ticks[0]} -Process {$_-$prev; $prev=$_} | Measure-Object -Minimum).Minimum)
    foreach($t in $ticks) {
        $line=[System.Windows.Forms.DataVisualization.Charting.StripLine]::new()
        $line.IntervalOffset=$t; $line.Interval=0; $line.BorderColor=Color 'grid'; $line.BorderWidth=2
        $axis.StripLines.Add($line)
        [void]$axis.CustomLabels.Add($t-$span,$t+$span,"$t%")
    }
}
function Set-RowAxis($axis,[string[]]$rows,[string[]]$colors) {
    $axis.Minimum=0.5; $axis.Maximum=$rows.Count+0.5
    for($i=0;$i -lt $rows.Count;$i++) {
        $label=$axis.CustomLabels.Add($rows.Count-$i-0.5,$rows.Count-$i+0.5,$rows[$i])
        $label.ForeColor=Color $(if($colors){$colors[$i]}else{'ink'})
    }
    $axis.LabelStyle.Font=Font 27
}
function Save-Chart($chart,[string]$name) {
    try { $chart.SaveImage((Join-Path $figureDir $name),[System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png) }
    finally { $chart.Dispose() }
}
# Horizontal bars in side-by-side panels sharing one set of row labels.
# Each panel: @{Title; Values; Labels; Highlight (bool[]); Ticks}
function Bar-Figure([string]$name,[string]$title,[string]$subtitle,[string[]]$rows,$panels,[int]$labelWidth,[int]$rowHeight=78) {
    $width=1600; $top=128; $headH=50; $axisH=48; $gap=64; $marginL=24; $marginR=24
    $height=$top+$headH+$rows.Count*$rowHeight+$axisH+16
    $chart=New-Canvas $width $height $title $subtitle
    $panelW=($width-$marginL-$marginR-$labelWidth-$gap*($panels.Count-1))/$panels.Count
    for($p=0;$p -lt $panels.Count;$p++) {
        $panel=$panels[$p]
        $plotX=$marginL+$labelWidth+$p*($panelW+$gap)
        $left=if($p -eq 0){$labelWidth}else{0}
        Add-Text $chart $panel.Title $plotX $top $panelW $headH (Font 27 -Bold) 'ink'
        $area=Add-Area $chart "p$p" ($plotX-$left) ($top+$headH) ($panelW+$left) ($rows.Count*$rowHeight+$axisH) $left $axisH
        # Leave ~130px past the last tick so outside value labels are never clipped.
        $last=$panel.Ticks[-1]
        Set-ValueAxis $area.AxisY 0 ($last*$panelW/($panelW-130)) $panel.Ticks
        if($p -eq 0) { Set-RowAxis $area.AxisX $rows } else {
            $area.AxisX.Minimum=0.5; $area.AxisX.Maximum=$rows.Count+0.5
            $area.AxisX.LabelStyle.Enabled=$false
        }
        $area.AxisX.LineWidth=2
        $s=[System.Windows.Forms.DataVisualization.Charting.Series]::new("s$p")
        $s.ChartArea="p$p"; $s.ChartType='Bar'; $s['PointWidth']='0.62'; $s['BarLabelStyle']='Outside'
        $s.Font=Font 26; $s.LabelForeColor=Color 'ink'
        for($i=0;$i -lt $rows.Count;$i++) {
            $idx=$s.Points.AddXY([double]($rows.Count-$i),[double]$panel.Values[$i])
            $pt=$s.Points[$idx]; $pt.Label=$panel.Labels[$i]
            $pt.Color=Color $(if($panel.Highlight[$i]){'accent'}else{'context'})
            if($panel.Highlight[$i]) { $pt.Font=Font 26 -Bold }
        }
        $chart.Series.Add($s)
    }
    Save-Chart $chart $name
}
function Fmt([double]$value) { $value.ToString('F2',$culture) }
function Best($values,[switch]$Lowest) {
    $target=if($Lowest){($values|Measure-Object -Minimum).Minimum}else{($values|Measure-Object -Maximum).Maximum}
    @($values | ForEach-Object { [math]::Abs($_-$target) -lt 1e-9 })
}
function Pick($items,$mask) { @(for($i=0;$i -lt $items.Count;$i++){ if($mask[$i]){$items[$i]} }) }
function Join-Thai([string[]]$items) { if($items.Count -le 1){$items -join ''}else{($items[0..($items.Count-2)] -join ', ')+' และ '+$items[-1]} }

# 1-3. Whisper CER on one shared 0-80% scale and, for voice sets, sentences transcribed exactly.
$asrShort=@{
    A0='Whisper-tiny สำเร็จรูป'; A1='tiny + ปรับภาษาไทย'; A2='Whisper-small สำเร็จรูป'
    A3='small + LoRA + gTTS'; A4='small + เสียงผู้ใช้ 72'; A5='small + เสียงผู้ใช้ 112'; A6='small + เสียงผู้ใช้ 112 + ฝึกต่อ'
}
$setText=@{voice_24='คำสั่งเสียงจริง 24 ประโยค'; voice_60='คำสั่งเสียงจริง 60 ประโยค'; FLEURS='ชุด FLEURS ภาษาไทย 1,021 เสียง'}
foreach($set in 'voice_24','voice_60','FLEURS') {
    $rows=@($asr | Where-Object test_set -eq $set | Sort-Object model_id)
    $ids=@($rows.model_id)
    $cer=@($rows | ForEach-Object { [double]$_.cer*100 })
    $cerBest=Best $cer -Lowest
    $cerIds=@(Pick $ids $cerBest)
    $cerMin=($cer|Measure-Object -Minimum).Minimum
    $panels=@(@{Title='CER (%) · ยิ่งต่ำยิ่งดี'; Values=$cer; Labels=@($cer|ForEach-Object{(Fmt $_)+'%'}); Highlight=$cerBest; Ticks=[double[]](0,20,40,60,80)})
    if($set -eq 'FLEURS') {
        $title="$(Join-Thai $cerIds) มี CER ต่ำสุดบนเสียงพูดทั่วไป ($(Fmt $cerMin)%)"
    } else {
        $exact=@($rows | ForEach-Object { [double]$_.exact_match*100 })
        $exactBest=Best $exact
        $exactIds=@(Pick $ids $exactBest)
        $top=$rows | Where-Object { $_.model_id -eq $exactIds[0] }
        $panels+=@{Title='ถอดถูกทั้งประโยค (%) · ยิ่งสูงยิ่งดี'; Values=$exact; Labels=@($rows|ForEach-Object{"$($_.exact_correct)/$($_.n)"}); Highlight=$exactBest; Ticks=[double[]](0,25,50,75,100)}
        $title=if(($cerIds -join ',') -eq ($exactIds -join ',')) {
            "$(Join-Thai $cerIds) ถอดเสียงได้ดีที่สุด: CER $(Fmt $cerMin)% และถูกทั้งประโยค $($top.exact_correct)/$($top.n)"
        } else {
            "$(Join-Thai $cerIds) มี CER ต่ำสุด ($(Fmt $cerMin)%) แต่ $(Join-Thai $exactIds) ถอดถูกทั้งประโยคมากกว่า ($($top.exact_correct)/$($top.n))"
        }
    }
    $labels=@($rows | ForEach-Object { "$($_.model_id)   $($asrShort[$_.model_id])" })
    Bar-Figure "results_asr_$set.png" $title "Whisper บน$($setText[$set]) · สีน้ำเงิน = ดีที่สุดในแต่ละตัวชี้วัด" $labels $panels 470
}

# 4. MASSIVE: Accuracy and Macro-F1 as side-by-side bar panels, the best model per metric highlighted.
$modelRows=@($intentMap.Values)
$current=@($intentMap.Keys | ForEach-Object { $k=$_; $massive | Where-Object { $_.training_set -eq 'I4' -and $_.model_key -eq $k } })
$acc=@($current|ForEach-Object{[double]$_.accuracy*100}); $f1=@($current|ForEach-Object{[double]$_.macro_f1*100})
$accIds=@(Pick $modelRows (Best $acc))
$accMax=($acc|Measure-Object -Maximum).Maximum
$spread=$accMax-($acc|Measure-Object -Minimum).Minimum
$bestName=if($accIds.Count -gt 1 -and @($accIds -like 'Logistic Regression*').Count -eq $accIds.Count){'Logistic Regression ทั้งสองแบบ'}else{Join-Thai $accIds}
$pct=[double[]](0,25,50,75,100)
Bar-Figure 'results_intent_massive.png' "$bestName มี Accuracy สูงสุด ($(Fmt $accMax)%) แต่ทุกโมเดลต่างกันไม่ถึง $([math]::Ceiling($spread)) จุด" `
    'โมเดลที่ฝึกด้วย I4 · ทดสอบบน MASSIVE ภาษาไทยและอังกฤษ · สีน้ำเงิน = สูงสุดในแต่ละตัวชี้วัด' $modelRows @(
    @{Title='Accuracy (%)'; Values=$acc; Labels=@($acc|ForEach-Object{(Fmt $_)+'%'}); Highlight=(Best $acc); Ticks=$pct},
    @{Title='Macro-F1 (%)'; Values=$f1; Labels=@($f1|ForEach-Object{(Fmt $_)+'%'}); Highlight=(Best $f1); Ticks=$pct}
) 520

# 5. Real voice commands: correct text and Whisper A5 text as side-by-side bar panels.
$n=[int]$latest.n
$ref=@($intentMap.Keys|ForEach-Object{[double]$latest.intent_accuracy.$_.reference*100})
$fromA5=@($intentMap.Keys|ForEach-Object{[double]$latest.intent_accuracy.$_.'models/whisper-small-th-v3'*100})
$refIds=@(Pick $modelRows (Best $ref)); $voiceIds=@(Pick $modelRows (Best $fromA5))
$bestCount=[int][math]::Round(($fromA5|Measure-Object -Maximum).Maximum*$n/100)
$voiceTitle=if(($refIds -join ',') -eq ($voiceIds -join ',')) {
    "$(Join-Thai $voiceIds) จำแนกถูกมากที่สุดทั้งจากข้อความที่ถูกต้องและจากเสียง ($bestCount/$n)"
} else {
    "$(Join-Thai $voiceIds) จำแนกคำสั่งจากเสียงถูกมากที่สุด ($bestCount/$n ประโยค)"
}
Bar-Figure 'results_intent_voice.png' $voiceTitle "คำสั่งเสียงจริง $n ประโยค · ฝึกด้วย I4 · สีน้ำเงิน = สูงสุดในแต่ละแผง" $modelRows @(
    @{Title="ข้อความที่ถูกต้อง ($n)"; Values=$ref; Labels=@($ref|ForEach-Object{"$([int][math]::Round($_*$n/100))/$n"}); Highlight=(Best $ref); Ticks=$pct},
    @{Title="ข้อความจาก Whisper A5 ($n)"; Values=$fromA5; Labels=@($fromA5|ForEach-Object{"$([int][math]::Round($_*$n/100))/$n"}); Highlight=(Best $fromA5); Ticks=$pct}
) 520

# 6. Whole system: three measures side by side, the best end-to-end configuration highlighted.
$sysBest=Best @($system|ForEach-Object{[double]$_.system_accuracy})
$chosen=@(Pick $system $sysBest)[0]
$pct=[double[]](0,50,100)
Bar-Figure 'results_system.png' "$($chosen.asr_id) + Naive Bayes ถูกทั้งระบบมากที่สุด: $($chosen.system_correct)/60 ประโยค ($(Fmt ($chosen.system_accuracy*100))%)" `
    'Whisper แต่ละรุ่นร่วมกับ Naive Bayes (I4) · สีน้ำเงิน = ชุดโมเดลที่ให้ผลทั้งระบบสูงสุด' `
    @($system|ForEach-Object{"Whisper $($_.asr_id)"}) @(
    @{Title='จำแนกคำสั่งถูก (60)'; Values=@($system|ForEach-Object{$_.intent_accuracy*100}); Labels=@($system|ForEach-Object{"$($_.intent_correct)/60"}); Highlight=$sysBest; Ticks=$pct},
    @{Title='คำสั่ง + รายละเอียดถูก (30)'; Values=@($system|ForEach-Object{$_.write_success_accuracy*100}); Labels=@($system|ForEach-Object{"$($_.write_success_correct)/30"}); Highlight=$sysBest; Ticks=$pct},
    @{Title='ถูกทั้งระบบ (60)'; Values=@($system|ForEach-Object{$_.system_accuracy*100}); Labels=@($system|ForEach-Object{"$($_.system_correct)/60"}); Highlight=$sysBest; Ticks=$pct}
) 210 84


$lines=[Collections.Generic.List[string]]::new()
foreach($line in @('# ข้อมูลสรุปผลสำหรับหัวข้อ 4','','ข้อมูลในหน้านี้อ่านจากผลทดลองที่บันทึกไว้ สคริปต์สร้างตาราง CSV และกราฟจากตัวเลขชุดเดียวกัน','','## ผลของ Whisper','','CER และ WER คืออัตราความผิดพลาด ยิ่งต่ำยิ่งดี ส่วนถูกทั้งประโยคคือข้อความตรงกับคำตอบหลังจัดรูปแบบ ยิ่งสูงยิ่งดี','','| โมเดล | ชุดทดสอบ | จำนวน | CER | WER | ถูกทั้งประโยค |','| --- | --- | ---: | ---: | ---: | ---: |')) { $lines.Add($line) }
foreach($r in $asr) {
    $exact=if($null -eq $r.exact_match){'ไม่ได้บันทึก'}else{"$($r.exact_correct)/$($r.n) ($(Percent $r.exact_match))"}
    $lines.Add("| $($r.model_id) — $($r.model) | $($r.test_set) | $($r.n) | $(Percent $r.cer) | $(Percent $r.wer) | $exact |")
}
foreach($line in @('','## ผลโมเดลจำแนกคำสั่งที่ฝึกด้วย I4','','| โมเดล | MASSIVE: Accuracy | MASSIVE: Macro-F1 | ข้อความที่ถูกต้อง (60) | เสียง A4 (60) | เสียง A5 (60) | เสียง A6 (60) |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |')) {$lines.Add($line)}
foreach($key in $intentMap.Keys) {
    $m=$current | Where-Object model_key -eq $key
    $v=$latest.intent_accuracy.$key
    $lines.Add("| $($intentMap[$key]) | $(Percent $m.accuracy) | $(Percent $m.macro_f1) | $(Percent $v.reference) | $(Percent $v.'models/whisper-small-th-v2') | $(Percent $v.'models/whisper-small-th-v3') | $(Percent $v.'models/whisper-small-th-v4') |")
}
foreach($line in @('','## ผลทั้งระบบ','','| Whisper + Naive Bayes | จำแนกถูก / 60 | รายละเอียดถูก / 30 | คำสั่งและรายละเอียดถูก / 30 | อ่านหรือคำสั่งอื่นถูก / 30 | ถูกทั้งระบบ / 60 |','| --- | ---: | ---: | ---: | ---: | ---: |')) {$lines.Add($line)}
foreach($r in $system) {$lines.Add("| $($r.asr_id) | $($r.intent_correct) | $($r.all_slots_correct) | $($r.write_success_correct) | $($r.read_correct) | $($r.system_correct) ($(Percent $r.system_accuracy)) |")}
foreach($line in @('','## แหล่งข้อมูลและการสร้างซ้ำ','','- `asr_results.csv`: ผล Whisper แยกตามชุดทดสอบ พร้อมไฟล์ต้นทาง','- `intent_massive_results.csv`: ผล MASSIVE ของโมเดลที่ฝึกด้วย I1–I4','- `intent_voice_results.csv`: ผลคำสั่งจริงทุกโมเดลและทุกชุดทดลอง','- `system_results.csv`: ผลจำแนก รายละเอียด และผลร่วมทั้งระบบ','- `system_per_command.csv`: ผลเพิ่มงาน ปิดงาน และเพิ่มนัดหมาย','- `results.json`: ข้อมูลทั้งหมดและหมายเหตุเกี่ยวกับการวัดผล','','สร้างข้อมูลและกราฟใหม่จากโฟลเดอร์หลักของโครงการ:','','```powershell','powershell -ExecutionPolicy Bypass -File scripts/build_results.ps1','```','','รหัส I1–I4 อ้างอิงการจับคู่รุ่นทดลองใน report.md เพราะ JSON เดิมไม่ได้เก็บรหัสชุดฝึกไว้ ผลต่างรุ่นจึงอาจมีทั้งข้อมูลฝึกและโมเดลถอดเสียงที่เปลี่ยนไป','','ตรวจจำนวนคำตอบถูกเทียบกับไฟล์ utterances_asrv4_60.csv และตรวจว่าข้อความกับผลจำแนกตรงกับรายละเอียดใน slots_asrv2–4.json ก่อนรวมผลทั้งระบบ','','ผลทั้งระบบวัดการเข้าใจคำสั่งและรายละเอียด โดยการปิดงานใช้รายการงานจำลอง ไม่ใช่อัตราความสำเร็จของการบันทึกข้อมูลจริงลง Google','','ชุดทดสอบมีผู้พูดคนเดียว และใช้ช่วยเลือกโมเดล จึงต้องประเมินซ้ำกับผู้พูดและข้อมูลใหม่')) {$lines.Add($line)}
Write-Utf8 (Join-Path $summaryDir 'README.md') ($lines -join "`n")
Write-Output "Verified and exported: ASR=$($asr.Count), MASSIVE=$($massive.Count), voice=$($voice.Count), system=$($system.Count); 6 charts."
