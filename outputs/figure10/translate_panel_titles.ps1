$ErrorActionPreference='Stop'
$dir='E:\Code\Thesis\agent_modeling\outputs\figure10'
$ppt=New-Object -ComObject PowerPoint.Application
$deck=$null
try {
  $deck=$ppt.Presentations.Open("$dir\图10_六分图_图片外可编辑.pptx",0,0,0)
  $s=$deck.Slides.Item(1)
  $titles=@{
    '分图a标题 可编辑'='(a) 图例检测（82个候选图例项）'
    '分图b标题 可编辑'='(b) 色彩提取（41种独立颜色）'
    '分图c标题 可编辑'='(c) 自适应区域分割（5129个多边形）'
    '分图d标题 可编辑'='(d) 边界强化与区域匹配'
    '分图e标题 可编辑'='(e) 图例属性表（6个采样项）'
    '分图f标题 可编辑'='(f) 核心定量指标'
  }
  foreach($o in $s.Shapes){
    if($titles.ContainsKey($o.Name)){
      $o.TextFrame.TextRange.Text=$titles[$o.Name]
      $o.TextFrame.TextRange.Font.NameFarEast='微软雅黑'
      $o.TextFrame.TextRange.Font.Size=13
      $o.TextFrame.TextRange.Font.Bold=-1
      $o.TextFrame.TextRange.ParagraphFormat.Alignment=2
    }
  }
  $deck.SaveAs("$dir\图10_中文子图标题_可编辑.pptx",24)
  $s.Export("$dir\图10_中文子图标题_预览.png",'PNG',2538,1526)
  Write-Output 'Updated all six panel titles.'
}finally{if($deck){$deck.Close()};$ppt.Quit()}
