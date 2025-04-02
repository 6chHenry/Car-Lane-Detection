# Car-Lane-detection
Original data comes from [GitHub](https://github.com/harshilp24/car-lane-detection/tree/master).

However, the code provided by the author doesn’t work well and even has some errors, such as calling the function improperly.  
So I’ve made changes to those codes, adding some creative functions:

1. I draw the region of interests by trial and error, finally make the convex region out.  
2. I improve the accuracy of detecting lanes by limiting the slope of lines, lowering the case of misjudging.  
3. I add blurring and dilating procedures to enhance the data.  

With effort, though, I haven’t achieved a consistent result yet. I will continue fixing the problems.

### 0402 Update:
I've written `cannytest.py` by myself. However, I can't apply it because it will run out of CPU when processing videos. I'm seeking solutions to it.  
I've also written `houghtransform.py`.
