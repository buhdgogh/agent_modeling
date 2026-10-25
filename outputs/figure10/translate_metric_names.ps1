$ErrorActionPreference='Stop'
$dir='E:\Code\Thesis\agent_modeling\outputs\figure10'
$ppt=New-Object -ComObject PowerPoint.Application
$deck=$null
try {
  $deck=$ppt.Presentations.Open("$dir\图10_中文子图标题_可编辑.pptx",0,0,0)
  $s=$deck.Slides.Item(1)
  $labels=@('检测图例项数','独立颜色数','分割区域数','区域—图例匹配数','图例覆盖率','匹配置信度','平均轮廓点数','分割方法')
  $count=0
  foreach($o in $s.Shapes){
    if($o.Name -match '^f 指标([1-8])名称 可编辑$'){
      $index=[int]$Matches[1]-1
      $o.TextFrame.TextRange.Text=$labels[$index]
      $o.TextFrame.TextRange.Font.NameFarEast='微软雅黑'
      $count++
    }
  }
  if($count -ne 8){throw "Expected 8 metric names, found $count"}
  $deck.SaveAs("$dir\图10_中文标题及指标_可编辑.pptx",24)
  $s.Export("$dir\图10_中文标题及指标_预览.png",'PNG',2538,1526)
  Write-Output "Updated $count metric names."
}finally{if($deck){$deck.Close()};$ppt.Quit()}
