$ErrorActionPreference='Stop'
# Reuse the established native PowerPoint helpers without executing its build.
$helpers=Get-Content -LiteralPath 'E:\Code\Thesis\agent_modeling\outputs\figure6\build_figure6.ps1' -Raw
$helpers=$helpers.Replace('.Line.Weight=$weight','.Line.Weight=[double]$weight')
Invoke-Expression ($helpers.Substring(0,$helpers.IndexOf('try {')))
$deck.PageSetup.SlideWidth=1268
$deck.PageSetup.SlideHeight=693
$province='C:\Users\zxt\AppData\Local\Temp\codex-clipboard-a2153a5e-b80b-4223-876b-042df3e13568.jpg'
$hangzhou='C:\Users\zxt\AppData\Local\Temp\codex-clipboard-a720e3e6-067a-4d99-bc91-0e7f1cb7a565.jpg'
function CroppedPicture($slide,$name,$path,$x,$y,$w,$h,$left,$top,$right,$bottom){
  $o=$slide.Shapes.AddPicture($path,0,-1,$x,$y,$w,$h)
  $o.Name=$name
  $o.LockAspectRatio=0
  $crop=$o.PictureFormat.Crop
  $crop.ShapeWidth=$w; $crop.ShapeHeight=$h
  $crop.PictureWidth=$w/($right-$left)
  $crop.PictureHeight=$h/($bottom-$top)
  $crop.PictureOffsetX=$crop.PictureWidth*(0.5-($left+$right)/2)
  $crop.PictureOffsetY=$crop.PictureHeight*(0.5-($top+$bottom)/2)
  $o.Left=$x; $o.Top=$y
  return $o
}
function Label($s,$name,$value,$x,$y,$w,$h,$size=18,$bold=$false,$color='000000',$align=1,$background=$false){
  $o=Text $s $name $value $x $y $w $h $size $bold $color $align
  if($background){$o.Fill.Solid();$o.Fill.ForeColor.RGB=Color 'FFFFFF';$o.Fill.Transparency=0.1}
  return $o
}
function NorthOriginal($s,$x,$y,$name){
  $a=Line $s "$name 指北箭头" $x ($y+30) $x $y '000000' 2.5
  $a.Line.EndArrowheadStyle=3
  $null=Label $s "$name 北向文字N" 'N' ($x-10) ($y+32) 25 28 23 $true '000000' 2 $true
}
try {
  $s=$deck.Slides.Add(1,12)
  $s.FollowMasterBackground=0;$s.Background.Fill.ForeColor.RGB=Color 'FFFFFF'
  $null=CroppedPicture $s '底图A 浙江省地质图 原始上传图片 可更换可调整裁剪' $province 44 38 502 490 (37/1440) (120/1726) (1333/1440) (1370/1726)
  $null=CroppedPicture $s '底图B 杭州幅 原始上传图片 可更换可调整裁剪' $hangzhou 667 38 562 434 (358/1984) (149/1271) (1595/1984) (1101/1271)
  $null=Rect $s 'A图外框 可编辑' 44 38 502 490 'none' '000000' 1.2
  $null=Rect $s 'B图外框 可编辑' 667 38 562 434 'none' '000000' 1.2
  $null=Label $s '分图A标题 可编辑' '(a) 图件在浙江省的位置' 56 1 505 32 24
  $null=Label $s '分图B标题 可编辑' '(b) 杭州幅与局部研究区' 679 1 560 32 24
  # The red locator box and dot reproduce the positions in the original Fig. 6.
  $null=Rect $s 'A图 杭州幅覆盖范围 红色实线框 可编辑' 248 133 96 75 'none' 'BC202F' 3
  $null=Label $s 'A图 杭州幅覆盖范围文字 可编辑' '杭州幅覆盖范围' 155 61 166 29 21 $false 'BC202F' 1 $true
  $null=Line $s 'A图 范围引线 可编辑' 232 93 247 132 'BC202F' 2
  $dot=$s.Shapes.AddShape(9,263,160,10,10);$dot.Name='A图 局部研究区位置红点 可编辑';$dot.Fill.ForeColor.RGB=Color 'BC202F';$dot.Line.Visible=0
  $null=Line $s 'A到B 上侧放大连线 可编辑' 344 133 667 38 '737373' 1.1 3
  $null=Line $s 'A到B 下侧放大连线 可编辑' 344 208 667 472 '737373' 1.1 3
  $null=Rect $s 'B图 杭州幅覆盖边框 红色实线 可编辑' 667 38 562 434 'none' 'BC202F' 2.2
  NorthOriginal $s 506 114 'A图'
  NorthOriginal $s 1179 88 'B图'
  # All exterior coordinate ticks and labels are native objects.
  $lonLabels=@('120°00′E','120°15′E','120°30′E','120°45′E','121°00′E')
  for($i=0;$i -lt 5;$i++){
    $x=667+$i*562/4
    $null=Line $s "B图 经度刻度$i 可编辑" $x 472 $x 479 '000000' 1
    $null=Label $s "B图 经度标签$i 可编辑" $lonLabels[$i] ($x-49) 482 100 25 17 $false '000000' 2
  }
  $latLabels=@('30°40′N','30°30′N','30°20′N','30°10′N','30°00′N')
  for($i=0;$i -lt 5;$i++){
    $y=38+$i*434/4
    $null=Line $s "B图 纬度刻度$i 可编辑" 660 $y 667 $y '000000' 1
    $null=Label $s "B图 纬度标签$i 可编辑" $latLabels[$i] 581 ($y-12) 76 24 17 $false '000000' 3
  }
  $rx=667+(120.16229882-120)*562
  $rw=(120.24444023-120.16229882)*562
  $ry=38+((30+40/60)-30.41457209)/(40/60)*434
  $rh=(30.41457209-30.35324517)/(40/60)*434
  $null=Rect $s 'B图 研究区经纬度包络框 黑色虚线 可编辑' $rx $ry $rw $rh 'none' '000000' 3 4
  $null=Label $s 'B图 局部研究区说明文字 可编辑' "局部研究区`n经纬度包络框" 850 115 128 53 21 $false '000000' 2 $true
  $arrow=Line $s 'B图 局部研究区指示箭头 可编辑' 874 171 ($rx+$rw+4) ($ry+$rh/2) '000000' 2
  $arrow.Line.EndArrowheadStyle=3
  $null=Label $s 'B图 比例尺文字 可编辑' '约10 km' 1010 411 74 27 19 $false '000000' 1 $true
  $scaleLength=10/(111.32*[Math]::Cos(30.33333*[Math]::PI/180))*562
  $null=Rect $s 'B图 10km比例尺黑色段 可编辑' 1013 440 ($scaleLength/2) 5 '000000' '000000' 0.5
  $null=Rect $s 'B图 10km比例尺白色段 可编辑' (1013+$scaleLength/2) 440 ($scaleLength/2) 5 'FFFFFF' '000000' 0.5
  $null=Label $s 'A图 底图说明 可编辑' '底图：浙江省地质图，1∶500000' 44 539 550 28 20
  $null=Label $s 'B图 底图说明 可编辑' '底图：杭州幅，1∶200000' 665 535 565 28 20
  $null=Label $s '研究区经度范围 可编辑' '研究区：120.1623°～120.2444°E' 665 566 565 28 20
  $null=Label $s '研究区纬度范围 可编辑' '　　　　30.3532°～30.4146°N' 665 595 565 28 20
  $null=Label $s '图例说明 可编辑' '红色实线：杭州幅覆盖范围；黑色虚线：局部研究区经纬度包络框（非实际边界）。' 44 634 1170 26 18
  $null=Label $s '近似定位说明 可编辑' '说明：按原图图框经纬网近似定位，非精确地理配准成果；不表示行政区界。' 44 663 1170 23 16 $false '666666'
  $s.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text=@'
图6 实验图件与局部研究区的空间位置：(a)杭州幅在浙江省的位置；(b)杭州幅覆盖范围与局部研究区经纬度包络框。
依据论文原始图6重新构建，使用用户本轮上传的浙江省地质图与杭州幅地质图作为两张底图；所有外加文字、坐标轴标签、刻度、红色框、黑色虚线框、指示箭头、放大连线、指北针、比例尺和说明文字均为PowerPoint原生可编辑对象。
底图内部印刷的地图文字、地质色块及线划属于图片像素，不转为可编辑对象。两张原始图片嵌入PPTX，可通过图片格式调整裁剪或更换。
沿用原图的近似定位与布局。右图外框为120—121 E、30—30°40′ N；研究区包络框为120.16229882—120.24444023 E、30.35324517—30.41457209 N。左图覆盖框与定位点沿用原图的近似位置。
原始图片1：codex-clipboard-a2153a5e-b80b-4223-876b-042df3e13568.jpg。
原始图片2：codex-clipboard-a720e3e6-067a-4d99-bc91-0e7f1cb7a565.jpg。
'@
  $final="$dir\图6_原图布局_底图外全部可编辑.pptx"
  $deck.SaveAs($final,24)
  $s.Export("$dir\图6_原图布局_预览.png",'PNG',2536,1386)
  Write-Output "Saved: $final"
  Write-Output "Total objects: $($s.Shapes.Count)"
} finally {
  if($deck){$deck.Close()};if($ppt){$ppt.Quit()}
}
