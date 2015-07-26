package KMeans;

import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


public class kmeans {

  public static class KmeansMapper 
      extends Mapper<LongWritable, TwoDPointWritable, IntWritable, TwoDPointWritable>{
      public final static String centerfile="centers.txt";
      public float[][] centroids = new float[2][2];

      public void setup(Context context) throws IOException {
	  Scanner reader = new Scanner(new FileReader(centerfile));

	  for (int  i=0; i<2; i++ ) {
	      int pos = reader.nextInt();

	      centroids[pos][0] = reader.nextFloat();
	      centroids[pos][1] = reader.nextFloat();
	  }
      }

      public void map(LongWritable key, TwoDPointWritable value, Context context
	  ) throws IOException, InterruptedException {

	  // Read number of centroids and current centroids from file
	  // calculate distance of given point to each Centroid
	  // winnerCentroid = centroid with minimarl distance for this point
	  float distance=0;
	  float mindistance=999999999.9f;
	  int winnercentroid=-1;
	  int i=0;
	  for ( i=0; i<2; i++ ) {
	      FloatWritable X = value.getx();
	      FloatWritable Y = value.gety();
	      float x = X.get();
	      float y = Y.get();
	      distance = ( x-centroids[i][0])*(x-centroids[i][0]) + 
		  (y - centroids[i][1])*(y-centroids[i][1]);
	      if ( distance < mindistance ) {
		  mindistance = distance;
		  winnercentroid=i;
	      }
	  }

	  IntWritable winnerCentroid = new IntWritable(winnercentroid);
	  context.write(winnerCentroid, value);
	  System.out.printf("Map: Centroid = %d distance = %f\n", winnercentroid, mindistance);
      }
  }

  
  public static class KmeansReducer 
      extends Reducer<IntWritable,TwoDPointWritable,IntWritable,Text> {

      public void reduce(IntWritable clusterid, Iterable<TwoDPointWritable> points, 
			 Context context
	  ) throws IOException, InterruptedException {

	  int num = 0;
	  float centerx=0.0f;
	  float centery=0.0f;
	  for (TwoDPointWritable point : points) {
	      num++;
	      FloatWritable X = point.getx();
	      FloatWritable Y = point.gety();
	      float x = X.get();
	      float y = Y.get();
	      centerx += x;
	      centery += y;
	  }
	  centerx = centerx/num;
	  centery = centery/num;
	  
	  String preres = String.format("%f %f", centerx, centery);
	  Text result = new Text(preres);
	  context.write(clusterid, result);
      }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    /*String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
    if (otherArgs.length != 2) {
      System.err.println("Usage: kmeans <in> <out>");
      System.exit(2);
    }*/

    Job job = new Job(conf, "kmeans");
    //Path toCache = new Path("/centers/centers.txt");
    //job.addCacheFile(toCache.toUri());
    //job.createSymlink();
		     
    job.setJarByClass(kmeans.class);
    job.setMapperClass(KmeansMapper.class);
    job.setReducerClass(KmeansReducer.class);

    job.setInputFormatClass (TwoDPointFileInputFormat.class);
    FileInputFormat.addInputPath(job, new Path(args[0]));

    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(TwoDPointWritable.class);

    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(Text.class);

    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}


