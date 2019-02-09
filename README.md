# Hyperspheres
I wrote these methods to try to generate a uniform spread of points on the surface of a sphere in any dimension. The display method became a fun way to visualize the distributions I was getting and to anime two-dimensional projections of three-dimensional casts of the hyperspheres. 
## Results
<p align="center">
  <img src="https://i.imgur.com/ga7arJn.gif" width="640" height="480" title="Collapsing casts">
</p>
This is a gif demonstrating a problem I was having with my first method of point generation. I was generating uniformly distributed points in a unit hypercube and then normalizing hem onto the surface of the unit hypersphere. This left a really strange cube atrifact in my distributions. Here I plot the first three axes of spheres (from dimensions 3 to 11) and color them as a 16x16x16 heatmap with brightness corresponding to a higher density in that voxel. As dimensionality increased, the distribution across any axis would get more uniform and more kurt (if that's a proper use of that word). Not sure why exactly that happened, but I feel like drawing and analogy of blowing up a balloon in a box is a good way to visualize what I was doing.

Will describe how I almost fixed the issue with an animation to go with it soon enough, but all my renders are in the MethodB folder anyway and the code is in the script. The second method leaves a stranfge octohedral(?) artefact in the sphere instead of a cube now but it's less dramatic.

I intend to add a third and fourth method soon, one that averages binary vectors and one (the big one) that simulated electrostatic repulsion withing the confines of the manifold until the distribution is as uniform as possible.
