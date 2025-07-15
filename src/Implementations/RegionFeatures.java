package Implementations;

import java.util.Arrays;
import java.util.List;
import graph.Node;
import graph.Region;

public class RegionFeatures {
	
	    private final Region region;

	    public RegionFeatures(Region region) {
	        this.region = region;
	    } 
	    public static double[] extract(Region region) {
	        return new double[] {
	                region.centroidX,
	                region.centroidY,
	                region.meanColor
	            };
	        }

	    // Extract features from region
	    public double[] extractFeatures() {
	        return new double[] {
	            region.meanColor,
	            region.centroidX,
	            region.centroidY
	        };
	    }

	    public static double computeSimilarity(Region r1, Region r2) {
	        return Math.abs(r1.meanColor - r2.meanColor);
	    }
	    public static void assignNormalizedFeaturesToNodes(List<Region> regions, List<Node> nodes) {
	        int featureLength = 3; // Example: meanColor, x, y
	        double[][] rawFeatures = new double[regions.size()][featureLength];

	     /// Extract raw features
	        for (int i = 0; i < regions.size(); i++) {
	            Region r = regions.get(i);
	            rawFeatures[i][0] = r.meanColor;               // Intensity
	            rawFeatures[i][1] = r.centroidX;               // X position
	            rawFeatures[i][2] = r.centroidY;               // Y position
	        }

	        // : Compute min and max per feature
	        double[] minVals = new double[featureLength];
	        double[] maxVals = new double[featureLength];
	        Arrays.fill(minVals, Double.MAX_VALUE);
	        Arrays.fill(maxVals, Double.MIN_VALUE);

	        for (double[] f : rawFeatures) {
	            for (int j = 0; j < featureLength; j++) {
	                minVals[j] = Math.min(minVals[j], f[j]);
	                maxVals[j] = Math.max(maxVals[j], f[j]);
	            }
	        }
// Normalize and assign
	        for (int i = 0; i < nodes.size(); i++) {
	            double[] norm = new double[featureLength];
	            for (int j = 0; j < featureLength; j++) {
	                if (maxVals[j] - minVals[j] == 0)
	                    norm[j] = 0.0;
	                else
	                    norm[j] = (rawFeatures[i][j] - minVals[j]) / (maxVals[j] - minVals[j]);
	            }
	            nodes.get(i).setFeatures(norm);
	        }
	    }

/*
    // CONFIGURABLE BIN SIZES
    private static final int H_BINS = 8;
    private static final int S_BINS = 4;
    private static final int V_BINS = 2;
    private static final int LBP_UNIFORM_PATTERNS = 59;
    
    public class Features{
    double[] ColorHistogram;
  //  double[] textureHistogram;
    double[] shapefeatures;
    double[] edgedensity;
    double[] position;
    
    
    public double[] getFullFeaturevector()
    {
    	return CombineFeatures(ColorHistogram,shapefeatures,shapefeatures,position);
    }
    
    public double[] CombineFeatures(double[]... arrays)
    {
    	int total=0;
    	 
         for (double[] arr : arrays) total += arr.length;
         double[] result = new double[total];
         int pos = 0;
         for (double[] arr : arrays) {
             System.arraycopy(arr, 0, result, pos, arr.length);
             pos += arr.length;
         }
         return result;
    	
    }
}
    
    public Features extractFeatures
    (BufferedImage image, List<Node> regionPixels) {
            double[] hsvHist = computeHSVHistogram(image, regionPixels);
            double[] shape = computeShapeFeatures(regionPixels);
            double[] position = computeNormalizedCentroid(regionPixels, image.getWidth(), image.getHeight());
            double[] edgeDensity = computeEdgeDensity(image, regionPixels);

            Features rf = new Features();
            rf.ColorHistogram = hsvHist;
            rf.shapefeatures = shape;
            rf.position = position;
            rf.edgedensity = edgeDensity;
            
            return rf;    
   }

    private double[] computeHSVHistogram(BufferedImage img, List<Node> nodes) {
        int bins = H_BINS * S_BINS * V_BINS;
        double[] hist = new double[bins];

        for (Node node : nodes) {
            int x = (int) node.getX();
            int y = (int) node.getY();
            Color c = new Color(img.getRGB(x, y));
            float[] hsv = Color.RGBtoHSB(c.getRed(), c.getGreen(), c.getBlue(), null);
            int hBin = Math.min((int)(hsv[0] * H_BINS), H_BINS - 1);
            int sBin = Math.min((int)(hsv[1] * S_BINS), S_BINS - 1);
            int vBin = Math.min((int)(hsv[2] * V_BINS), V_BINS - 1);
            int index = hBin * S_BINS * V_BINS + sBin * V_BINS + vBin;
            hist[index]++;
        }
        return normalize(hist);
    }

    private double[] computeShapeFeatures(List<Node> nodes) {
        int area = nodes.size();

        int minX = Integer.MAX_VALUE, maxX = Integer.MIN_VALUE;
        int minY = Integer.MAX_VALUE, maxY = Integer.MIN_VALUE;
        for (Node node : nodes) {
            int x = (int) node.getX();
            int y = (int) node.getY();
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }
        int width = maxX - minX + 1;
        int height = maxY - minY + 1;
        double aspectRatio = height == 0 ? 0 : (double) width / height;

        return new double[] { area, aspectRatio };
    }

    private double[] computeNormalizedCentroid(List<Node> nodes, int width, int height) {
        double sumX = 0, sumY=0;
        for (Node node : nodes) {
            sumX += node.getX();
            sumY += node.getY();
        }
        double cx = sumX / nodes.size();
        double cy = sumY / nodes.size();
        return new double[] { cx / width, cy / height };
    }

    private double[] computeEdgeDensity(BufferedImage img, List<Node> nodes) {
        int edgeCount = 0;
        for (Node node : nodes) {
            int x = (int) node.getX();
            int y = (int) node.getY();
            if (x <= 0 || y <= 0 || x >= img.getWidth() - 1 || y >= img.getHeight() - 1)
                continue;

            int gx = sobel(img, x - 1, y, true);
            int gy = sobel(img, x, y - 1, false);
            double mag = Math.sqrt(gx * gx + gy * gy);
            if (mag > 100) edgeCount++;
        }
        double density = (double) edgeCount / nodes.size();
        return new double[] { density };
    }

    private int sobel(BufferedImage img, int x, int y, boolean isX) {
        int[][] kernelX = {{-1,0,1},{-2,0,2},{-1,0,1}};
        int[][] kernelY = {{-1,-2,-1},{0,0,0},{1,2,1}};
        int[][] kernel = isX ? kernelX : kernelY;
        int sum = 0;
        for (int i=0;i<3;i++) {
            for (int j=0;j<3;j++) {
                int px = x + j -1;
                int py = y + i -1;
                if (px<0 || py<0 || px>=img.getWidth() || py>=img.getHeight()) continue;
                int val = new Color(img.getRGB(px,py)).getRed();
                sum += val * kernel[i][j];
            }
        }
        return sum;
    }

    private double[] normalize(double[] arr) {
        double sum = Arrays.stream(arr).sum();
        if (sum == 0) return arr;
        return Arrays.stream(arr).map(x -> x / sum).toArray();
    }


    	*/	

}
