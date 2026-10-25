$ErrorActionPreference = 'Stop'
$dir = 'E:\Code\Thesis\agent_modeling\outputs\figure6'
$source = 'E:\Code\Thesis\agent_modeling\tmp_review_1025\image6_source.png'
$ppt = New-Object -ComObject PowerPoint.Application
$deck = $ppt.Presentations.Add(0)
$deck.PageSetup.SlideWidth = 960
$deck.PageSetup.SlideHeight = 560
function Color($hex) { return [Convert]::ToInt32($hex.Substring(0,2),16) + 256*[Convert]::ToInt32($hex.Substring(2,2),16) + 65536*[Convert]::ToInt32($hex.Substring(4,2),16) }
function Text($s,$name,$value,$x,$y,$w,$h,$size=13,$bold=$false,$color='222222',$align=1) {
  $o=$s.Shapes.AddTextbox(1,$x,$y,$w,$h); $o.Name=$name
  $o.TextFrame.MarginLeft=0; $o.TextFrame.MarginRight=0; $o.TextFrame.MarginTop=0; $o.TextFrame.MarginBottom=0
  $o.TextFrame.WordWrap=0; $o.TextFrame.TextRange.Text=$value
  $o.TextFrame.TextRange.Font.Name='Arial'; $o.TextFrame.TextRange.Font.NameFarEast='微软雅黑'
  $o.TextFrame.TextRange.Font.Size=$size; $o.TextFrame.TextRange.Font.Bold=([int]$bold)*-1
  $o.TextFrame.TextRange.Font.Color.RGB=Color $color; $o.TextFrame.TextRange.ParagraphFormat.Alignment=$align
  return $o
}
function Rect($s,$name,$x,$y,$w,$h,$fill='none',$stroke='222222',$weight=1,$dash=1) {
  $o=$s.Shapes.AddShape(1,$x,$y,$w,$h); $o.Name=$name
  if($fill -eq 'none'){$o.Fill.Visible=0}else{$o.Fill.Solid();$o.Fill.ForeColor.RGB=Color $fill}
  if($stroke -eq 'none'){$o.Line.Visible=0}else{$o.Line.ForeColor.RGB=Color $stroke;$o.Line.Weight=$weight;$o.Line.DashStyle=$dash}
  return $o
}
function Line($s,$name,$x1,$y1,$x2,$y2,$color='444444',$weight=1,$dash=1) {
  $o=$s.Shapes.AddLine($x1,$y1,$x2,$y2);$o.Name=$name;$o.Line.ForeColor.RGB=Color $color;$o.Line.Weight=$weight;$o.Line.DashStyle=$dash;return $o
}
function North($s,$x,$y,$tag) {
  $o=Line $s "$tag 指北针" $x ($y+25) $x $y '222222' 1.5; $o.Line.EndArrowheadStyle=3
  $null=Text $s "$tag 北向标记" 'N' ($x-10) ($y+28) 20 20 13 $true '222222' 2
}
function Scale($s,$x,$y,$len,$halfLabel,$endLabel,$tag) {
  $null=Rect $s "$tag 比例尺黑段" $x $y ($len/2) 5 '222222' '222222' 0.6
  $null=Rect $s "$tag 比例尺白段" ($x+$len/2) $y ($len/2) 5 'FFFFFF' '222222' 0.6
  $null=Text $s "$tag 比例尺0" '0' ($x-5) ($y+8) 15 16 10
  $null=Text $s "$tag 比例尺中点" $halfLabel ($x+$len/2-15) ($y+8) 30 16 10 $false '222222' 2
  $null=Text $s "$tag 比例尺终点" $endLabel ($x+$len-23) ($y+8) 55 16 10 $false '222222' 2
}
try {
  $s=$deck.Slides.Add(1,12);$s.FollowMasterBackground=0;$s.Background.Fill.ForeColor.RGB=Color 'FFFFFF'
  $null=Text $s '分图a标题' '(a) 杭州幅在浙江省的位置' 38 15 400 30 18 $true
  $null=Text $s '分图b标题' '(b) 杭州幅覆盖范围与局部研究区' 480 15 455 30 18 $true
  # Left locator uses the source province outline only as a generalized locator.
  $ax=39.0;$ay=58.0;$aw=356.0;$ah=386.0
  $null=Rect $s 'a 定位图背景' $ax $ay $aw $ah 'F2F7FA' '919BA3' 0.7
  $geo=Get-Content -LiteralPath "$dir\zhejiang_boundary.json" -Raw | ConvertFrom-Json
  $sx=($aw-28)/(123.25-117.9);$sy=$sx/[Math]::Cos(29.5*[Math]::PI/180)
  function PX($lon){return $ax+14+($lon-117.9)*$sx}
  function PY($lat){return $ay+($ah-4.35*$sy)/2+(31.3-$lat)*$sy}
  $polyIndex=0
  foreach($poly in $geo.features[0].geometry.coordinates){
    $ring=$poly[0]; if($ring.Count -lt 4){continue}
    $xs=@($ring|ForEach-Object {$_[0]}); $ys=@($ring|ForEach-Object {$_[1]})
    $spanX=($xs|Measure-Object -Maximum -Minimum);$spanY=($ys|Measure-Object -Maximum -Minimum)
    if(($spanX.Maximum-$spanX.Minimum)*($spanY.Maximum-$spanY.Minimum) -lt 0.0015){continue}
    $polyIndex++; $ff=$s.Shapes.BuildFreeform(0,(PX $ring[0][0]),(PY $ring[0][1]))
    for($j=1;$j -lt $ring.Count;$j++){$ff.AddNodes(0,0,(PX $ring[$j][0]),(PY $ring[$j][1]))}
    $o=$ff.ConvertToShape();$o.Name="浙江省轮廓 $polyIndex 可编辑顶点";$o.Fill.Solid();$o.Fill.ForeColor.RGB=Color 'E4E8DF';$o.Line.ForeColor.RGB=Color '59695A';$o.Line.Weight=0.6
  }
  $null=Text $s '浙江省标注' '浙 江 省' 125 264 170 35 23 $true '65705F' 2
  $null=Text $s '东海标注' '东 海' 313 297 65 25 15 $false '6F95AD' 2
  $lx=PX 120;$ly=PY (30+40/60);$lw=(PX 121)-$lx;$lh=(PY 30)-$ly
  $null=Rect $s 'a 杭州幅覆盖范围 红色实线' $lx $ly $lw $lh 'none' 'B42132' 1.8
  $null=Text $s 'a 杭州幅文字' '杭州幅' 90 82 95 25 15 $true 'B42132'
  $null=Line $s 'a 杭州幅引线' 160 107 ($lx+3) ($ly+5) 'B42132' 1.1
  $cx=PX 120.20335;$cy=PY 30.3839
  $dot=$s.Shapes.AddShape(9,($cx-2),($cy-2),4,4);$dot.Name='a 局部研究区位置';$dot.Fill.ForeColor.RGB=Color '222222';$dot.Line.Visible=0
  North $s 368 78 'a'
  Scale $s 60 404 (100/(111.32*[Math]::Cos(29.5*[Math]::PI/180))*$sx) '50' '100 km' 'a'
  # Right extent frame uses the exact coordinates printed on the supplied figure.
  $bx=496.0;$by=93.0;$bw=418.0;$bh=323.0
  function BX($lon){return $bx+($lon-120)*$bw}
  function BY($lat){return $by+((30+40/60)-$lat)/(40/60)*$bh}
  $null=Rect $s 'b 杭州幅内部' $bx $by $bw $bh 'FAFAF5' 'none'
  foreach($lon in @(120.25,120.5,120.75)){$null=Line $s "b 经线 $lon" (BX $lon) $by (BX $lon) ($by+$bh) 'D7DEDF' 0.6 4}
  foreach($lat in @((30+10/60),(30+20/60),(30+30/60))){$null=Line $s "b 纬线 $lat" $bx (BY $lat) ($bx+$bw) (BY $lat) 'D7DEDF' 0.6 4}
  # Extent enlargement connectors run behind the frame and labels.
  $null=Line $s '两图上放大连线' ($lx+$lw) $ly $bx $by '90989C' 0.8 4
  $null=Line $s '两图下放大连线' ($lx+$lw) ($ly+$lh) $bx ($by+$bh) '90989C' 0.8 4
  $null=Rect $s 'b 杭州幅覆盖范围 红色实线' $bx $by $bw $bh 'none' 'B42132' 1.8
  $lonTicks=@('120°00′E','120°15′E','120°30′E','120°45′E','121°00′E')
  for($i=0;$i -le 4;$i++){$x=$bx+$i*$bw/4;$null=Line $s "b 经度刻度$i" $x ($by+$bh) $x ($by+$bh+5) '222222' 0.8;$null=Text $s "b 经度标签$i" $lonTicks[$i] ($x-35) ($by+$bh+9) 75 20 11 $false '222222' 2}
  $latTicks=@('30°40′N','30°30′N','30°20′N','30°10′N','30°00′N')
  for($i=0;$i -le 4;$i++){$y=$by+$i*$bh/4;$null=Line $s "b 纬度刻度$i" ($bx-5) $y $bx $y '222222' 0.8;$null=Text $s "b 纬度标签$i" $latTicks[$i] ($bx-66) ($y-8) 58 18 11 $false '222222' 3}
  $rx=BX 120.16229882;$ry=BY 30.41457209;$rw=(BX 120.24444023)-$rx;$rh=(BY 30.35324517)-$ry
  $null=Rect $s 'b 局部研究区经纬度包络框 非实际边界' $rx $ry $rw $rh 'E4B465' '222222' 1.8 4
  $null=Text $s 'b 局部研究区标注' "局部研究区`n经纬度包络框" 643 157 155 50 16 $true
  $arrow=Line $s 'b 包络框指示箭头' 649 208 ($rx+$rw+3) ($ry+$rh/2) '222222' 1.2;$arrow.Line.EndArrowheadStyle=3
  $null=Text $s 'b 杭州幅标注' '杭州幅覆盖范围' 641 110 195 28 16 $false 'B42132' 2
  North $s 879 118 'b'
  $kmPerDegree=111.32*[Math]::Cos(30.3333*[Math]::PI/180)
  Scale $s 767 370 (20/$kmPerDegree*$bw) '10' '20 km' 'b'
  $null=Text $s '研究区经度范围' '研究区：120.1623°—120.2444°E' 496 455 420 23 13
  $null=Text $s '研究区纬度范围' '　　　　30.3532°—30.4146°N' 496 479 420 23 13
  $null=Line $s '图例杭州幅线型' 40 477 73 477 'B42132' 1.8
  $null=Text $s '图例杭州幅文字' '杭州幅覆盖范围' 82 466 160 25 13
  $null=Rect $s '图例研究区框' 40 501 33 15 'E4B465' '222222' 1.5 4
  $null=Text $s '图例研究区文字' '研究区经纬度包络框' 82 494 230 25 13
  $null=Text $s '定位图说明' '定位示意；省界已概化；包络框不代表研究区实际边界。' 39 532 880 20 11 $false '666666'
  $s.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text=@'
图6 实验图件与局部研究区的空间位置：(a)杭州幅在浙江省的位置；(b)杭州幅覆盖范围与局部研究区经纬度包络框。
本页为全矢量可编辑定位示意图。所有省界轮廓、坐标标签、图例、范围框、指北针和比例尺均为PowerPoint原生形状或文本。
研究区坐标依据用户论文：120.16229882—120.24444023 E，30.35324517—30.41457209 N。
杭州幅覆盖范围依据原图6的坐标框：120—121 E，30—30°40′ N；该范围沿用原图的近似定位表达，未独立核验图幅四至。
浙江省轮廓来源：https://geo.datav.aliyun.com/areas_v3/bound/330000.json ，访问日期2026-10-25。仅作概化定位示意，不用于测绘或行政边界认定；未将该轮廓作为研究区WGS84精确配准依据。
比例尺按当地平均纬度近似计算。右图仅展示空间位置，不表达地质单元和岩性。
原图底图注明杭州幅1:200000，正文注明1:50000，两者不一致，需要作者核实。因此本页不标称源图比例尺。
'@
  # Preserve the author's original figure as a separate visual reference.
  $s2=$deck.Slides.Add(2,12);$s2.FollowMasterBackground=0;$s2.Background.Fill.ForeColor.RGB=Color 'FFFFFF'
  $null=Text $s2 '原图参考标题' '原始图6参考' 36 12 500 25 16 $true
  $null=$s2.Shapes.AddPicture($source,0,-1,22,44,916,500.7)
  $s2.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text='本页保留用户DOCX中的原始图6作为参考，为嵌入位图。请使用第1页的原生矢量定位图进行编辑。原图来源：论文-1022-updated-中文版本.docx，word/media/image6.png。'
  $deck.SaveAs("$dir\图6_空间位置_可编辑.pptx",24)
  $s.Export("$dir\图6_预览.png",'PNG',1920,1120)
  $s2.Export("$dir\原始图6_参考预览.png",'PNG',1920,1120)
  Write-Output "Saved $dir\图6_空间位置_可编辑.pptx"
  Write-Output "Editable slide shape count: $($s.Shapes.Count)"
} finally {
  if($deck){$deck.Close()};if($ppt){$ppt.Quit()}
}
