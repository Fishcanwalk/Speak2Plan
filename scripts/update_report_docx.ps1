param(
    [int]$Section = 4,
    [int]$EndSection = $Section,
    [string]$Root = (Split-Path $PSScriptRoot -Parent)
)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.Drawing
$target=Join-Path $Root 'reports/report.docx'
$markdown=Get-Content -LiteralPath (Join-Path $Root 'reports/report.md') -Raw -Encoding UTF8
$sectionText=[regex]::Match($markdown,"(?ms)^## $Section\..*?(?=^## |\z)").Value
if(!$sectionText) { throw "Markdown section $Section missing" }
$zip=[IO.Compression.ZipFile]::OpenRead($target)
function Read-Part([string]$name) {
    $entry=$zip.GetEntry($name)
    if(!$entry) { throw "DOCX part missing: $name" }
    $reader=[IO.StreamReader]::new($entry.Open(),[Text.Encoding]::UTF8)
    try { return $reader.ReadToEnd() } finally {$reader.Dispose()}
}
[xml]$doc=Read-Part 'word/document.xml'
[xml]$rels=Read-Part 'word/_rels/document.xml.rels'
[xml]$types=Read-Part '[Content_Types].xml'
$w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
$rns='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
$ns=[Xml.XmlNamespaceManager]::new($doc.NameTable); $ns.AddNamespace('w',$w)
$body=$doc.SelectSingleNode('//w:body',$ns)
function Para-Text($node) { return (($node.SelectNodes('.//w:t',$ns) | ForEach-Object InnerText) -join '') }
$start=$null; $stop=$null
foreach($node in $body.ChildNodes) {
    if($node.LocalName -ne 'p') {continue}
    $text=Para-Text $node
    if(!$start -and $text -match "^$Section\. ") {$start=$node;continue}
    if($start -and $text -match '^(\d+)\. ' -and [int]$matches[1] -gt $EndSection) {$stop=$node;break}
    if($start) {
        $style=$node.SelectSingleNode('./w:pPr/w:pStyle',$ns)
        if($style -and $style.GetAttribute('val',$w) -eq 'Heading2' -and $text -notmatch '^\d+\. ') {
            $stop=$node;break
        }
    }
}
if(!$start -or !$stop){$zip.Dispose();throw 'DOCX section boundaries missing'}
$outside=@();$within=$false
foreach($node in $body.ChildNodes) {
    if($node -eq $start){$within=$true}
    if($node -eq $stop){$within=$false}
    if(!$within){$outside+=,$node.OuterXml}
}
$nextDrawing=1
foreach($node in $doc.SelectNodes('//*[local-name()="docPr"]')) {
    $nextDrawing=[math]::Max($nextDrawing,[int]$node.GetAttribute('id')+1)
}
$sec=$body.SelectSingleNode('./w:sectPr',$ns)
$page=$sec.SelectSingleNode('./w:pgSz',$ns);$margins=$sec.SelectSingleNode('./w:pgMar',$ns)
$usable=[int]$page.GetAttribute('w',$w)-[int]$margins.GetAttribute('left',$w)-[int]$margins.GetAttribute('right',$w)
if($usable -le 0){$usable=9360}
$imageParts=@{}
function Element([string]$name) { return $doc.CreateElement('w',$name,$w) }
function Val-Element([string]$name,[string]$value) {
    $node=Element $name; [void]$node.SetAttribute('val',$w,$value); return $node
}
function Append-Run($paragraph,[string]$text,[bool]$bold=$false,[bool]$code=$false) {
    if(!$text){return}
    $run=Element 'r'
    if($bold -or $code) {
        $pr=Element 'rPr'
        if($bold){[void]$pr.AppendChild((Element 'b'))}
        if($code){$fonts=Element 'rFonts';[void]$fonts.SetAttribute('ascii',$w,'Consolas');[void]$fonts.SetAttribute('hAnsi',$w,'Consolas');[void]$pr.AppendChild($fonts)}
        [void]$run.AppendChild($pr)
    }
    $t=Element 't'
    $space=$doc.CreateAttribute('xml','space','http://www.w3.org/XML/1998/namespace')
    $space.Value='preserve';[void]$t.Attributes.Append($space);$t.InnerText=$text
    [void]$run.AppendChild($t);[void]$paragraph.AppendChild($run)
}
function Append-Inline($paragraph,[string]$text) {
    $position=0
    $pattern='\*\*(?<bold>.+?)\*\*|`(?<code>[^`]+)`|\[(?<label>[^\]]+)\]\((?<url>[^)]+)\)'
    foreach($match in [regex]::Matches($text,$pattern)) {
        Append-Run $paragraph $text.Substring($position,$match.Index-$position)
        if($match.Groups['bold'].Success){Append-Run $paragraph $match.Groups['bold'].Value $true}
        elseif($match.Groups['code'].Success){Append-Run $paragraph $match.Groups['code'].Value $false $true}
        else{Append-Run $paragraph $match.Groups['label'].Value; Append-Run $paragraph " ($($match.Groups['url'].Value))"}
        $position=$match.Index+$match.Length
    }
    Append-Run $paragraph $text.Substring($position)
}
function Paragraph([string]$text,[string]$style='BodyText') {
    $p=Element 'p';$pr=Element 'pPr';[void]$pr.AppendChild((Val-Element 'pStyle' $style));[void]$p.AppendChild($pr)
    Append-Inline $p $text
    return $p
}
function Picture([string]$relative,[string]$caption) {
    $path=Join-Path (Join-Path $Root 'reports') $relative
    if(!(Test-Path -LiteralPath $path)){throw "Figure missing: $relative"}
    $name=Split-Path $relative -Leaf; $part="word/media/$name"
    $script:imageParts[$part]=$path
    $relation=$rels.DocumentElement.SelectSingleNode("*[local-name()='Relationship' and @Target='media/$name']")
    if($relation){$rid=$relation.GetAttribute('Id')}
    else {
        $number=1
        $ids=@($rels.DocumentElement.ChildNodes | ForEach-Object {$_.GetAttribute('Id')})
        while($ids -contains "rIdResults$number") {$number++}
        $rid="rIdResults$number"
        $relation=$rels.CreateElement('Relationship',$rels.DocumentElement.NamespaceURI)
        $relation.SetAttribute('Id',$rid);$relation.SetAttribute('Type',"$rns/image");$relation.SetAttribute('Target',"media/$name")
        [void]$rels.DocumentElement.AppendChild($relation)
    }
    $image=[Drawing.Image]::FromFile($path)
    try {$cx=[long]($usable*635);$cy=[long]($cx*$image.Height/$image.Width)} finally {$image.Dispose()}
    $id=$script:nextDrawing; $script:nextDrawing++
    $safeName=[Security.SecurityElement]::Escape($name);$safeCaption=[Security.SecurityElement]::Escape($caption)
    $fragment=$doc.CreateDocumentFragment()
    $fragment.InnerXml=@"
<w:p xmlns:w="$w"><w:r><w:drawing><wp:inline xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" distT="0" distB="0" distL="0" distR="0"><wp:extent cx="$cx" cy="$cy"/><wp:docPr id="$id" name="$safeName" descr="$safeCaption"/><wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="0" name="$safeName"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip xmlns:r="$rns" r:embed="$rid"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="$cx" cy="$cy"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>
"@
    return $fragment.FirstChild
}
function Table($lines) {
    $table=Element 'tbl';$properties=Element 'tblPr';$width=Element 'tblW'
    [void]$width.SetAttribute('w',$w,[string]$usable);[void]$width.SetAttribute('type',$w,'dxa');[void]$properties.AppendChild($width)
    $borders=Element 'tblBorders'
    foreach($edge in 'top','left','bottom','right','insideH','insideV') {
        $border=Element $edge;[void]$border.SetAttribute('val',$w,'single');[void]$border.SetAttribute('sz',$w,'4');[void]$border.SetAttribute('color',$w,'D5DCE5');[void]$borders.AppendChild($border)
    }
    [void]$properties.AppendChild($borders);[void]$table.AppendChild($properties)
    $columnCount=$lines[0].Trim().Trim('|').Split('|').Count
    $grid=Element 'tblGrid'
    for($column=0;$column -lt $columnCount;$column++) {
        $gridColumn=Element 'gridCol'
        $fraction=if($column -eq 0){0.28}else{0.72/($columnCount-1)}
        [void]$gridColumn.SetAttribute('w',$w,[string][int]($usable*$fraction));[void]$grid.AppendChild($gridColumn)
    }
    [void]$table.AppendChild($grid)
    $index=0
    foreach($line in $lines) {
        if($line -match '^\|[\s:|\-]+\|$'){continue}
        $cells=$line.Trim().Trim('|').Split('|')
        $row=Element 'tr';$rowPr=Element 'trPr';[void]$rowPr.AppendChild((Element 'cantSplit'))
        if($index -eq 0){[void]$rowPr.AppendChild((Element 'tblHeader'))}
        [void]$row.AppendChild($rowPr)
        for($column=0;$column -lt $cells.Count;$column++) {
            $cell=Element 'tc';$cellPr=Element 'tcPr';$cw=Element 'tcW'
            $fraction=if($column -eq 0){0.28}else{0.72/($cells.Count-1)}
            [void]$cw.SetAttribute('w',$w,[string][int]($usable*$fraction));[void]$cw.SetAttribute('type',$w,'dxa');[void]$cellPr.AppendChild($cw)
            if($index -eq 0){$shading=Element 'shd';[void]$shading.SetAttribute('fill',$w,'EAF0FA');[void]$cellPr.AppendChild($shading)}
            [void]$cell.AppendChild($cellPr)
            $p=Paragraph $cells[$column].Trim()
            foreach($run in $p.SelectNodes('./w:r',$ns)) {
                $pr=$run.SelectSingleNode('./w:rPr',$ns)
                if(!$pr){$pr=Element 'rPr';[void]$run.PrependChild($pr)}
                [void]$pr.AppendChild((Val-Element 'sz' '18'))
                if($index -eq 0){[void]$pr.AppendChild((Element 'b'))}
            }
            [void]$cell.AppendChild($p);[void]$row.AppendChild($cell)
        }
        [void]$table.AppendChild($row);$index++
    }
    return $table
}
$newNodes=@();$lines=$sectionText -split '\r?\n'
for($i=0;$i -lt $lines.Count;$i++) {
    $line=$lines[$i]
    if(!$line.Trim()){continue}
    if($line.StartsWith('|')) {
        $tableLines=@()
        while($i -lt $lines.Count -and $lines[$i].StartsWith('|')){$tableLines+=,$lines[$i];$i++}
        $i--; $newNodes+=,(Table $tableLines)
    }
    elseif($line -match '^!\[(.*?)\]\((.*?)\)$') {
        $newNodes+=,(Picture $matches[2] $matches[1]);$newNodes+=,(Paragraph $matches[1] 'ImageCaption')
    }
    elseif($line -match '^(#{2,4}) (.+)$') {
        $style='Heading'+$matches[1].Length;$newNodes+=,(Paragraph $matches[2] $style)
    }
    else {
        if($line.StartsWith('- ')){$line='• '+$line.Substring(2)}
        $newNodes+=,(Paragraph $line)
    }
}
$node=$start
while($node -ne $stop){$next=$node.NextSibling;[void]$body.RemoveChild($node);$node=$next}
foreach($node in $newNodes){[void]$body.InsertBefore($node,$stop)}
# Confirm the existing sections outside the replacement are byte-identical at XML-node level.
$remaining=@($body.ChildNodes | Where-Object {$newNodes -notcontains $_} | ForEach-Object OuterXml)
if(($outside -join '') -cne ($remaining -join '')){$zip.Dispose();throw 'Unrelated section content changed'}
if(!$types.DocumentElement.SelectSingleNode('*[local-name()="Default" and @Extension="png"]')) {
    $type=$types.CreateElement('Default',$types.DocumentElement.NamespaceURI);$type.SetAttribute('Extension','png');$type.SetAttribute('ContentType','image/png');[void]$types.DocumentElement.AppendChild($type)
}
$replacement=@{'word/document.xml'=$doc.OuterXml;'word/_rels/document.xml.rels'=$rels.OuterXml;'[Content_Types].xml'=$types.OuterXml}
$temp=Join-Path ([IO.Path]::GetTempPath()) ('speak2plan_docx_'+[guid]::NewGuid().ToString('N')+'.docx')
$output=[IO.File]::Open($temp,[IO.FileMode]::CreateNew)
$archive=[IO.Compression.ZipArchive]::new($output,[IO.Compression.ZipArchiveMode]::Create,$false)
try {
    foreach($entry in $zip.Entries) {
        if($imageParts.ContainsKey($entry.FullName)){continue}
        $newEntry=$archive.CreateEntry($entry.FullName,[IO.Compression.CompressionLevel]::Optimal);$stream=$newEntry.Open()
        try {
            if($replacement.ContainsKey($entry.FullName)){$bytes=[Text.Encoding]::UTF8.GetBytes($replacement[$entry.FullName]);$stream.Write($bytes,0,$bytes.Length)}
            else{$input=$entry.Open();try{$input.CopyTo($stream)}finally{$input.Dispose()}}
        } finally {$stream.Dispose()}
    }
    foreach($part in $imageParts.Keys) {
        $newEntry=$archive.CreateEntry($part,[IO.Compression.CompressionLevel]::Optimal);$stream=$newEntry.Open();$input=[IO.File]::OpenRead($imageParts[$part])
        try{$input.CopyTo($stream)}finally{$input.Dispose();$stream.Dispose()}
    }
} finally {$archive.Dispose();$output.Dispose();$zip.Dispose()}
$check=[IO.Compression.ZipFile]::OpenRead($temp)
try {
    foreach($part in $replacement.Keys) {
        $reader=[IO.StreamReader]::new($check.GetEntry($part).Open());try{[xml]$null=$reader.ReadToEnd()}finally{$reader.Dispose()}
    }
    foreach($part in $imageParts.Keys){if(!$check.GetEntry($part)){throw "Missing embedded chart $part"}}
} finally {$check.Dispose()}
Move-Item -LiteralPath $temp -Destination $target -Force
Write-Output "Updated section $Section; embedded $($imageParts.Count) charts; other sections preserved."
