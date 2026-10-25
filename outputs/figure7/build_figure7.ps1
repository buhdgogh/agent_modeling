$ErrorActionPreference='Stop'
$helpers=Get-Content -LiteralPath 'E:\Code\Thesis\agent_modeling\outputs\figure6\build_figure6.ps1' -Raw
$helpers=$helpers.Replace('.Line.Weight=$weight','.Line.Weight=[double]$weight')
Invoke-Expression ($helpers.Substring(0,$helpers.IndexOf('try {')))
$dir='E:\Code\Thesis\agent_modeling\outputs\figure7'
$deck.PageSetup.SlideWidth=1267
$deck.PageSetup.SlideHeight=424
function T($slide,$name,$value,$x,$y,$w,$h,$size=13,$bold=$false,$color='111111',$align=1){
  return (Text $slide $name $value $x $y $w $h $size $bold $color $align)
}
function Tag($slide,$name,$value,$x,$y,$w,$h,$fill,$stroke,$fontColor='172127',$size=14,$bold=$true){
  $o=Rect $slide $name $x $y $w $h $fill $stroke 0.6
  $o.TextFrame.MarginLeft=3;$o.TextFrame.MarginRight=3;$o.TextFrame.MarginTop=1;$o.TextFrame.MarginBottom=0
  $o.TextFrame.VerticalAnchor=3;$o.TextFrame.WordWrap=0
  $o.TextFrame.TextRange.Text=$value
  $o.TextFrame.TextRange.Font.Name='Arial';$o.TextFrame.TextRange.Font.NameFarEast='微软雅黑'
  $o.TextFrame.TextRange.Font.Size=$size;$o.TextFrame.TextRange.Font.Bold=([int]$bold)*-1
  $o.TextFrame.TextRange.Font.Color.RGB=Color $fontColor
  $o.TextFrame.TextRange.ParagraphFormat.Alignment=2
  return $o
}
function Code($slide,$name,$value,$x,$y,$w,$h){
  $o=Tag $slide $name $value $x $y $w $h 'FFF5DC' 'F9E7B4' '111111' 10 $false
  $o.TextFrame.TextRange.Font.Name='Consolas'
  $o.TextFrame.TextRange.ParagraphFormat.Alignment=1
  $o.TextFrame.MarginLeft=4
  return $o
}
try {
  $s=$deck.Slides.Add(1,12);$s.FollowMasterBackground=0;$s.Background.Fill.ForeColor.RGB=Color 'FFFFFF'
  # Native independent backgrounds and text, arranged to match the source image.
  foreach($i in 0..3){$null=Line $s "步骤$($i+1)到$($i+2)时序连接线" 72 (129+$i*64) 72 (136+$i*64) '87969E' 0.8}
  foreach($i in 0..4){$fill=if($i%2 -eq 0){'F7F7F7'}else{'FFFFFF'};$null=Rect $s "第$($i+1)步行背景" 21 (71+$i*64) 1224 58 $fill 'E5E8EB' 0.65}
  $null=Tag $s '列标题 Step' 'Step' 63 25 39 21 '37474F' '37474F' 'FFFFFF' 14
  $null=Tag $s '列标题 节点' '节点' 182 25 42 21 '37474F' '37474F' 'FFFFFF' 14
  $null=Tag $s '列标题 Model' 'Model' 325 25 49 21 '37474F' '37474F' 'FFFFFF' 14
  $null=Tag $s '列标题 执行内容' '执行内容' 475 25 72 21 '37474F' '37474F' 'FFFFFF' 14
  $null=Tag $s '列标题 State Record' 'State Record' 1005 25 93 21 '37474F' '37474F' 'FFFFFF' 13
  foreach($i in 0..4){$null=T $s "步骤编号$($i+1)" ([string]($i+1)) 57 (91+$i*64) 30 24 16 $true '37474F' 2}
  $null=Tag $s '步骤1 决策节点' '决策节点' 161 90 67 20 'EEF4F8' 'D8E6ED'
  $null=Tag $s '步骤2 工具执行器' '工具执行器' 145 154 83 20 'E9F4F0' 'D4E4DE'
  $null=T $s '步骤2 并行说明' '（并行）' 235 170 58 18 11 $true '168EFF'
  $null=Tag $s '步骤3 决策节点' '决策节点' 161 218 67 20 'EEF4F8' 'D8E6ED'
  $null=Tag $s '步骤4 工具执行器' '工具执行器' 145 282 83 20 'E9F4F0' 'D4E4DE'
  $null=Tag $s '步骤5 冲突检查器' '冲突检查器' 145 346 83 20 'FFF5DC' 'F9E7B4'
  $null=T $s '步骤1 模型' 'Qwen-Max' 299 94 160 20 12
  $null=T $s '步骤2 主模型' 'Qwen-Max' 299 157 160 20 12
  $null=T $s '步骤2 图件模型' '+ qwen-vl-max' 299 173 160 18 10 $false '65717A'
  $null=T $s '步骤3 无API调用' '(no API call)' 299 222 160 20 12
  $null=T $s '步骤4 融合模型' 'Qwen-Max(fusion)' 285 285 180 20 12
  $null=T $s '步骤4 空间计算智能体' '+ GempyAgent' 298 301 165 18 10 $false '65717A'
  $null=T $s '步骤5 规则引擎' '(rule engine)' 299 350 160 20 12
  $null=T $s '步骤1 执行内容' '语义解析：识别全模态输入，规划两步处理流程' 476 94 515 24 13
  $null=T $s '步骤2 执行内容' '并行执行：文本（31个地层，41个剖面）+ 图件（82个图例，5129个区域）' 476 158 520 24 13
  $null=T $s '步骤3 执行内容' '状态检查：已完成文本与图件处理，转入流水线' 476 222 515 24 13
  $null=T $s '步骤4 执行内容' '生成融合CSV，随后进行空间计算（202条钻孔记录）' 476 286 515 24 13
  $null=T $s '步骤5 执行内容' '四维检查：文本12项与图件12项，无冲突，检查通过' 476 350 515 24 13
  $null=Code $s '步骤1 状态记录' 'completed_tools = []' 1005 92 126 17
  $null=Code $s '步骤2 状态记录' "+ extract_text_info_tool`n+ analyze_image_tool" 1005 151 157 29
  $null=Code $s '步骤3 状态记录' '(unchanged)' 1005 220 73 17
  $null=Code $s '步骤4 状态记录' '+ auto_borehole_pipeline' 1005 284 151 17
  $null=Code $s '步骤5 状态记录' "has_conflicts = False`nconfidence = 0.78" 1005 343 138 29
  $null=T $s '图下注释' 'Step 3: state-check only, no LLM call.  Step 2: text & image agents run concurrently via ThreadPoolExecutor.' 36 397 1200 20 11 $false '92979C'
  $s.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text=@'
图7 多模态建模任务的智能体调度流程。
根据用户提供的论文-1022-updated-中文版本.docx中的图7（word/media/image7.png）重建。保留原图的五步结构、列顺序、文字和数值；图内所有文字、标签、背景框与步骤连接线均为PowerPoint原生可编辑对象，没有将原图作为背景或合并位图。
原图第4步写“202条钻孔记录”，论文当前正文在虚拟钻孔生成和三维建模部分写“66个虚拟钻孔位置、132个地层界面点”。本图按原图保留202，需作者核对是否为不同运行或不同统计口径，再决定是否修改。
原图第2步记录31个地层、41个剖面、82个图例、5129个区域；第5步记录文本12项与图件12项以及confidence=0.78，均照原图保留，不重新解释其统计口径。
'@
  $file="$dir\图7_智能体调度流程_全部可编辑.pptx"
  $deck.SaveAs($file,24)
  $s.Export("$dir\图7_预览.png",'PNG',2534,848)
  Write-Output "Saved: $file"
  Write-Output "Native objects: $($s.Shapes.Count)"
} finally {if($deck){$deck.Close()};if($ppt){$ppt.Quit()}}
