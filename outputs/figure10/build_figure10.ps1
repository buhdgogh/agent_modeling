$ErrorActionPreference='Stop'
$helpers=Get-Content -LiteralPath 'E:\Code\Thesis\agent_modeling\outputs\figure6\build_figure6.ps1' -Raw
$helpers=$helpers.Replace('.Line.Weight=$weight','.Line.Weight=[double]$weight')
Invoke-Expression ($helpers.Substring(0,$helpers.IndexOf('try {')))
$dir='E:\Code\Thesis\agent_modeling\outputs\figure10'
$source="$dir\figure10_source.png"
$deck.PageSetup.SlideWidth=1269
$deck.PageSetup.SlideHeight=763
function T($s,$name,$value,$x,$y,$w,$h,$size=12,$bold=$false,$color='000000',$align=1){
  $o=Text $s $name $value $x $y $w $h $size $bold $color $align
  $o.TextFrame.AutoSize=0;$o.Left=[double]$x;$o.Top=[double]$y;$o.Width=[double]$w;$o.Height=[double]$h
  $o.TextFrame.TextRange.ParagraphFormat.Alignment=$align
  return $o
}
function CropImage($s,$name,$x,$y,$w,$h,$l,$t,$r,$b){
  $o=$s.Shapes.AddPicture($source,0,-1,$x,$y,$w,$h);$o.Name=$name;$o.LockAspectRatio=0
  $c=$o.PictureFormat.Crop;$c.ShapeWidth=$w;$c.ShapeHeight=$h
  $c.PictureWidth=$w/(($r-$l)/1269);$c.PictureHeight=$h/(($b-$t)/763)
  $c.PictureOffsetX=$c.PictureWidth*(0.5-($l+$r)/2538)
  $c.PictureOffsetY=$c.PictureHeight*(0.5-($t+$b)/1526)
  $o.Left=$x;$o.Top=$y;return $o
}
try {
  $s=$deck.Slides.Add(1,12);$s.FollowMasterBackground=0;$s.Background.Fill.ForeColor.RGB=Color 'FFFFFF'
  $null=T $s '主标题 可编辑 图号已修正' 'Fig. 10. Geological map image extraction pipeline and quantitative results' 30 7 1209 21 15 $true '000000' 2
  $null=T $s '副标题 可编辑' '(Hangzhou 1:200K map; YOLOv10 detection + qwen-vl-max parsing + dual-track segmentation)' 20 27 1229 21 15 $true '000000' 2
  $null=T $s '分图a标题 可编辑' '(a) Legend Detection (82 items)' 10 123 405 23 13 $true '000000' 2
  $null=T $s '分图b标题 可编辑' '(b) Color Extraction (41 unique colors)' 430 102 409 23 13 $true '000000' 2
  $null=T $s '分图c标题 可编辑' '(c) Region Segmentation (5,129 polygons)' 850 123 405 23 13 $true '000000' 2
  $null=T $s '分图d标题 可编辑' '(d) Boundary Reinforcement & Matching' 10 455 405 23 13 $true '000000' 2
  $null=T $s '分图e标题 可编辑' '(e) Legend Attribute Table (excerpt)' 430 435 409 23 13 $true '000000' 2
  $null=T $s '分图f标题 可编辑' '(f) Quantitative Metrics' 851 435 405 23 13 $true '000000' 2
  $null=CropImage $s 'a YOLOv10图例检测结果 图像可替换可裁剪' 10 144 405 257 10 144 415 401
  $null=CropImage $s 'c 自适应区域分割结果 图像可替换可裁剪' 850 144 405 257 850 144 1255 401
  $null=CropImage $s 'd 边界强化与区域匹配结果 图像可替换可裁剪' 10 475 405 258 10 475 415 733
  $colors=@(
    @{x=488;y=127;rgb='RGB(241,204,178)';hex='#f1ccb2';fill='F1CCB2'},
    @{x=672;y=127;rgb='RGB(255,158,195)';hex='#ff9ec3';fill='FF9EC3'},
    @{x=488;y=248;rgb='RGB(0,251,1)';hex='#00fb01';fill='00FB01'},
    @{x=672;y=248;rgb='RGB(240,12,12)';hex='#f00c0c';fill='F00C0C'}
  )
  for($i=0;$i -lt $colors.Count;$i++){
    $d=$colors[$i]
    $o=Rect $s "b 代表性色块$($i+1) 可编辑填色" $d.x $d.y 122 72 $d.fill '111111' 1
    $null=T $s "b 色块$($i+1) RGB数值 可编辑" $d.rgb ($d.x-10) ($d.y+75) 142 15 10 $true '000000' 2
    $null=T $s "b 色块$($i+1) HEX编码 可编辑" $d.hex ($d.x-10) ($d.y+89) 142 15 9 $false '505050' 2
  }
  # Editable native PowerPoint table, with independent native color swatches.
  $rows=@(
    @('ID','Color','RGB','Y(px)'),
    @('26','','(241,204,178)','1300'),
    @('10','','(255,158,195)','2956'),
    @('48','','(151,254,136)','4608'),
    @('27','','(255,183,135)','6264'),
    @('67','','(0,0,0)','7915'),
    @('43','','(240,7,7)','9569')
  )
  $tableShape=$s.Shapes.AddTable(7,4,430,475,409,252)
  $tableShape.Name='e 六个采样图例属性 原生可编辑表格'
  $table=$tableShape.Table
  $table.Columns.Item(1).Width=51;$table.Columns.Item(2).Width=112;$table.Columns.Item(3).Width=158;$table.Columns.Item(4).Width=88
  foreach($ri in 1..7){
    $table.Rows.Item($ri).Height=36
    foreach($ci in 1..4){
      $cell=$table.Cell($ri,$ci)
      $cell.Shape.Fill.Solid();$cell.Shape.Fill.ForeColor.RGB=Color 'FFFFFF'
      foreach($bi in 1..4){$cell.Borders.Item($bi).ForeColor.RGB=Color 'FFFFFF';$cell.Borders.Item($bi).Weight=0.25}
      $tf=$cell.Shape.TextFrame;$tf.MarginLeft=2;$tf.MarginRight=2;$tf.MarginTop=0;$tf.MarginBottom=0;$tf.VerticalAnchor=3
      $tf.TextRange.Text=$rows[$ri-1][$ci-1];$tf.TextRange.Font.Name='Arial';$tf.TextRange.Font.Size=10;$tf.TextRange.Font.Bold=0;$tf.TextRange.Font.Color.RGB=Color '000000';$tf.TextRange.ParagraphFormat.Alignment=2
      if($ri -eq 1){$tf.TextRange.Font.Bold=-1;$cell.Borders.Item(3).ForeColor.RGB=Color '000000';$cell.Borders.Item(3).Weight=0.6}
    }
  }
  $null=Line $s 'e 属性表表头下分隔线 可编辑' 430 511 839 511 '000000' 0.6
  $rowFills=@('F1CCB2','FF9EC3','97FE88','FFB787','000000','F00707')
  foreach($i in 0..5){$null=Rect $s "e 第$($i+1)行RGB色块 可编辑填色" 524 (518+$i*36) 49 22 $rowFills[$i] 'none'}
  $null=T $s 'e 地层顺序说明 可编辑' 'Top to bottom: younger to older strata' 440 738 390 18 10 $false '555555' 2
  $metrics=@(
    @('Legend items detected','82'),
    @('Independent colors','41'),
    @('Segmented regions','5,129'),
    @('Region–Legend matches','5,129'),
    @('Legend coverage','74.4% (61/82)'),
    @('Match confidence','0.95'),
    @('Avg contour points','12.9 (simplified)'),
    @('Segmentation method','YOLOv10-seg + Manhattan')
  )
  foreach($i in 0..7){
    $null=T $s "f 指标$($i+1)名称 可编辑" $metrics[$i][0] 864 (494+$i*30.3) 177 20 10 $true
    $null=T $s "f 指标$($i+1)数值 可编辑" $metrics[$i][1] 1048 (494+$i*30.3) 214 20 10 $true '0055C8'
  }
  $s.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text=@'
图10 地质图件解析流程及定量结果。
源文件：论文-1022-updated-中文版本.docx，word/media/image10.png。
原图顶部Fig. 8已按论文中的实际编号修正为Fig. 10。沿用原图的六分图布局和英文标题。
(a)、(c)、(d)为从原图保留的三个图像面板，仅作PowerPoint图片裁剪，不生成或重绘其地质内容；图像内部已印刷的地图文字、检测框、分割线和像素不转换为可编辑对象。
其余主副标题、分图标题、(b)4个色块、RGB与HEX文字、(e)原生属性表及色块、(f)全部指标均为可编辑原生对象。
(b)展示四个代表性色块；(e)为不同采样项的RGB记录，两处红色RGB值不同，按原图分别保留。
(e) ID和Y(px)顺序沿用原图，地层由新到老的解释依据用户本次提供的图注；表中没有补造未给出的地层名称。
(f)各数值照原图保留。74.4%=61/82四舍五入至一位小数。Match confidence=0.95保留原图标签，不改写为独立验证准确率。
原图副标题使用Hangzhou 1:200K，与当前正文“1:5万”存在比例尺差异，本次保留原图标注，需作者统一核对。
'@
  $file="$dir\图10_六分图_图片外可编辑.pptx"
  $deck.SaveAs($file,24)
  $s.Export("$dir\图10_预览.png",'PNG',2538,1526)
  Write-Output "Saved: $file"
  Write-Output "Total objects: $($s.Shapes.Count)"
} finally {if($deck){$deck.Close()};if($ppt){$ppt.Quit()}}
