c append sb 
k cc append sb 
newChunk 
f length files f f 
font f getFont 
c charExists getBaseFont font 
cc k isSurrogatePair Utilities 
c getType Character FORMAT Character 
currentFont font 
cc k convertToUtf32 Utilities u 
length sb currentFont 
f length files f f 
newChunk toString sb currentFont Chunk 
font f getFont 
setLength sb 
u charExists getBaseFont font 
u getType Character FORMAT Character 
currentFont font 
currentFont font 
length sb currentFont 
newChunk toString sb currentFont Chunk 
setLength sb 
Chunk processChar cc k sb StringBuffer DocumentException IOException 
newChunk Chunk 
k cc c 
c c 
c append sb 
font Font 
c append sb 
currentFont font 
