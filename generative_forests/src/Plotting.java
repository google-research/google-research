// Companion Code to the paper "Generative Forests" by R. Nock and M. Guillame-Bert.

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Vector;
import javax.imageio.ImageIO;

class WeightedRectangle2D implements Debuggable {
    Rectangle2D.Double rectangle;
    double weight, density; // weight = unnormalized (just in case it is useful)

    public static double NUMERICAL_PRECISION_ERROR;

    WeightedRectangle2D(Rectangle2D.Double r, double w){
	rectangle = r;
	weight = w;
	density = -1.0;
    }

    WeightedRectangle2D(Rectangle2D.Double r, double w, double d){
	this(r, w);
	density = d;
    }

    public double surface(){
	return (rectangle.width * rectangle.height);
    }
    
    public boolean strictly_contains(double x, double y){
	// handling numerical precision errors
	if ( (x > rectangle.x + NUMERICAL_PRECISION_ERROR) && (x < rectangle.x + rectangle.width - NUMERICAL_PRECISION_ERROR) && (y > rectangle.y + NUMERICAL_PRECISION_ERROR) && (y < rectangle.y + rectangle.height - NUMERICAL_PRECISION_ERROR) )
	    return true;
	return false;
    }
    
    public boolean contains_X(double v){
	// handling numerical precision errors
	if ( (v >= rectangle.x - (NUMERICAL_PRECISION_ERROR/1000.0)) && (v <= rectangle.x + rectangle.width + (NUMERICAL_PRECISION_ERROR/1000.0)) )
	    return true;
	return false;
    }

    public boolean contains_Y(double v){
	// handling numerical precision errors
	if ( (v >= rectangle.y - (NUMERICAL_PRECISION_ERROR/1000.0)) && (v <= rectangle.y + rectangle.height + (NUMERICAL_PRECISION_ERROR/1000.0)) )
	    return true;
	return false;
    }
    
    public boolean matches(WeightedRectangle2D r){
	return (rectangle.equals(r.rectangle));
    }

    public String toString(){
	return rectangle + "{" + weight + ":" + density + "}";
    }
}

public class Plotting implements Debuggable{

    // class managing all plotting stuff

    public static int IMAGE_SIZE_MAX = 500;
    public static boolean SQUARE_IMAGE = true;
    public static boolean EMBED_DATA = true;
    public static int DATA_SIZE = 4;
    public static double OFFSET = 1;

    public static int GRID_SIZE = 101;
  public static String LOWER_LEFT = "LOWER_LEFT",
      UPPER_LEFT = "UPPER_LEFT",
      LOWER_RIGHT = "LOWER_RIGHT",
      UPPER_RIGHT = "UPPER_RIGHT";
    public static String LINES = "LINES", RECTANGLES = "RECTANGLES";
    
    int feature_index_x, feature_index_y, width, height;
    double x_min, x_max, y_min, y_max, length_x, length_y;
    
    Vector <Line2D.Double> all_lines;
    Vector <WeightedRectangle2D> all_rectangles;

    public static double GET_MIN(Dataset ds, int feature_index){
	if (Feature.IS_CONTINUOUS(ds.features.elementAt(feature_index).type))
	    return ds.features.elementAt(feature_index).dmin;
	else Dataset.perror("Plotting.class :: plotting non continuous variables");

	return -1.0;
    }

    public static double GET_FEATURE_VALUE(Dataset ds, Observation o, int feature_index){
	if (Feature.IS_CONTINUOUS(ds.features.elementAt(feature_index).type))
	    return o.typed_features.elementAt(feature_index).dv;
	else Dataset.perror("Plotting.class :: plotting non continuous variables");

	return -1.0;
    }
    
    public static double GET_MAX(Dataset ds, int feature_index){
	if (Feature.IS_CONTINUOUS(ds.features.elementAt(feature_index).type))
	    return ds.features.elementAt(feature_index).dmax;
	else Dataset.perror("Plotting.class :: plotting non continuous variables");

	return -1.0;
    }
    
    public static double GET_MAX(Feature f, double dmax){
	double d;
	if (Feature.IS_CONTINUOUS(f.type))
	    return f.dmax;
	else
	    Dataset.perror("Plotting.class :: plotting non continuous variables");

	return -1.0;
    }
    
    public static double GET_MIN(Feature f, double dmin){
	double d;
	if (Feature.IS_CONTINUOUS(f.type))
	    return f.dmin;
	else
	    Dataset.perror("Plotting.class :: plotting non continuous variables");

	return -1.0;
    }

    public static double MAP_COORDINATE_TO_IMAGE(double c, int size, double vmin, double vmax){
	double vuse;
	
	if ( (c < vmin) && (c > vmin - WeightedRectangle2D.NUMERICAL_PRECISION_ERROR) )
	    vuse = vmin;
	else if ( (c > vmax) && (c < vmax + WeightedRectangle2D.NUMERICAL_PRECISION_ERROR) )
	    vuse = vmax;
	else
	    vuse = c;

    if ((c <= vmin - WeightedRectangle2D.NUMERICAL_PRECISION_ERROR)
        || (c >= vmax + WeightedRectangle2D.NUMERICAL_PRECISION_ERROR))
      Dataset.perror(
          "Plotting.class :: value out of bounds ("
              + c
              + " not in ["
              + vmin
              + ","
              + vmax
              + "] +/- "
              + WeightedRectangle2D.NUMERICAL_PRECISION_ERROR
              + ")");

	return  (((vuse - vmin)/(vmax - vmin)) * ((double) size));
    }

    public static double MAP_X_TO_IMAGE(double c, int size, double vmin, double vmax){
	return MAP_COORDINATE_TO_IMAGE(c, size, vmin, vmax);
    }
    
    public static double MAP_Y_TO_IMAGE(double c, int size, double vmin, double vmax){
	return ((double) size - MAP_COORDINATE_TO_IMAGE(c, size, vmin, vmax));
    }

    Plotting(){
	feature_index_x = feature_index_y = width = height = width = height = -1;
	x_min = x_max = y_min = y_max = length_x = length_y = -1.0;
	all_lines = null;
	all_rectangles = null;
    }

    public void init_vectors(){
	all_lines = new Vector<>();
	all_rectangles = new Vector<>();
    }

    public void init(GenerativeModelBasedOnEnsembleOfTrees geot, int f_index_x, int f_index_y){
	feature_index_x = f_index_x;
	feature_index_y = f_index_y;
		
	x_min = GET_MIN(geot.myDS, feature_index_x);
	x_max = GET_MAX(geot.myDS, feature_index_x);
	y_min = GET_MIN(geot.myDS, feature_index_y);
	y_max = GET_MAX(geot.myDS, feature_index_y);

	length_x = x_max - x_min;
	length_y = y_max - y_min;

	if ( (SQUARE_IMAGE) || (length_x == length_y) ){
	    width = height = IMAGE_SIZE_MAX;
	}else{
	    if (length_x > length_y){
		width = IMAGE_SIZE_MAX;
		height = (int) ((length_y / length_x) * (double) IMAGE_SIZE_MAX);
	    }else{
		width = (int) ((length_x / length_y) * (double) IMAGE_SIZE_MAX);		
		height = IMAGE_SIZE_MAX;
	    }
	}
    }

    public void compute_geometric_objects(GenerativeModelBasedOnEnsembleOfTrees geot, String which_object){ //Plotting.LINES

	if ( (Feature.IS_NOMINAL(geot.myDS.features.elementAt(feature_index_x).type)) || (Feature.IS_NOMINAL(geot.myDS.features.elementAt(feature_index_y).type)) )
	    Dataset.perror("Plotting.class :: plotting non continuous / integer variables");

	double xmic, xmac, ymic, ymac, xc, yc;

	if (length_x < length_y)
	    WeightedRectangle2D.NUMERICAL_PRECISION_ERROR = length_x / 10000.0;
	else
	    WeightedRectangle2D.NUMERICAL_PRECISION_ERROR = length_y / 10000.0;

	MeasuredSupportAtTupleOfNodes init_mds = new MeasuredSupportAtTupleOfNodes(geot, true, false, -1), cur_mds;
	init_mds.generative_support.local_empirical_measure.init_proportions(1.0);
	    
	Vector <MeasuredSupportAtTupleOfNodes> m = new Vector<>();
	m.addElement(init_mds);

	init_vectors();
	GenerativeModelBasedOnEnsembleOfTrees.ALL_SUPPORTS_WITH_POSITIVE_MEASURE_GET(m, this, which_object);
    }
    
    public void add_line(MeasuredSupportAtTupleOfNodes cur_mds){
	Line2D.Double ld;
	double xmic, xmac, ymic, ymac, xc, yc;
	
	xmic = Plotting.MAP_X_TO_IMAGE(Plotting.GET_MIN(cur_mds.generative_support.support.feature(feature_index_x), x_min), width, x_min, x_max);
	xmac = Plotting.MAP_X_TO_IMAGE(Plotting.GET_MAX(cur_mds.generative_support.support.feature(feature_index_x), x_max), width, x_min, x_max);
	ymic = Plotting.MAP_Y_TO_IMAGE(Plotting.GET_MIN(cur_mds.generative_support.support.feature(feature_index_y), y_min), height, y_min, y_max);
	ymac = Plotting.MAP_Y_TO_IMAGE(Plotting.GET_MAX(cur_mds.generative_support.support.feature(feature_index_y), y_max), height, y_min, y_max);
	    
	ld = new Line2D.Double(xmic, ymic, xmac, ymic);
	if (!all_lines.contains(ld))
	    all_lines.add(ld);
	
	ld = new Line2D.Double(xmac, ymic, xmac, ymac);
	if (!all_lines.contains(ld))
	    all_lines.add(ld);
	
	ld = new Line2D.Double(xmic, ymac, xmac, ymac);
	if (!all_lines.contains(ld))
	    all_lines.add(ld);
	
	ld = new Line2D.Double(xmic, ymic, xmic, ymac);
	if (!all_lines.contains(ld))
	    all_lines.add(ld);
    }

     public void add_rectangle(MeasuredSupportAtTupleOfNodes cur_mds){
	double xmic, xmac, ymic, ymac, su;
	
	xmic = Plotting.GET_MIN(cur_mds.generative_support.support.feature(feature_index_x), x_min);
	xmac = Plotting.GET_MAX(cur_mds.generative_support.support.feature(feature_index_x), x_max);
	ymic = Plotting.GET_MIN(cur_mds.generative_support.support.feature(feature_index_y), y_min);
	ymac = Plotting.GET_MAX(cur_mds.generative_support.support.feature(feature_index_y), y_max);

	if ( (cur_mds.generative_support.local_empirical_measure.observations_indexes == null) || (cur_mds.generative_support.local_empirical_measure.observations_indexes.length == 0) )
	    su = 0.0;
	else{
	    su = cur_mds.generative_support.local_empirical_measure.total_weight; 
	}

	all_rectangles.addElement(new WeightedRectangle2D(new Rectangle2D.Double(xmic, ymic, xmac - xmic, ymac - ymic), su));
    }
    
    public void compute_and_store_x_y_frontiers(GenerativeModelBasedOnEnsembleOfTrees geot, String name_file){
	// stores the splits of the geot in the (x,y) plane and => image

	int i;
	double xc, yc;
	
	if ( (Feature.IS_NOMINAL(geot.myDS.features.elementAt(feature_index_x).type)) || (Feature.IS_NOMINAL(geot.myDS.features.elementAt(feature_index_y).type)) )
	    Dataset.perror("Plotting.class :: plotting non continuous / integer variables");

	if (length_x < length_y)
	    WeightedRectangle2D.NUMERICAL_PRECISION_ERROR = length_x / 10000.0;
	else
	    WeightedRectangle2D.NUMERICAL_PRECISION_ERROR = length_y / 10000.0;

	compute_geometric_objects(geot, Plotting.LINES);

	BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
	Graphics2D graph = (Graphics2D) img.getGraphics();

	graph.setPaint(Color.white);
	graph.fill(new Rectangle2D.Double(0,0,width,height));
	graph.setPaint(Color.black);
          
        graph.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);  
	for (i=0;i<all_lines.size();i++)
	    graph.draw(all_lines.elementAt(i));

	Observation o;
	if (EMBED_DATA){
	    graph.setPaint(Color.green);  
	    for (i=0;i<geot.myDS.observations_from_file.size();i++){
		o = geot.myDS.observations_from_file.elementAt(i);
		if ( (!Observation.FEATURE_IS_UNKNOWN(o, feature_index_x)) && (!Observation.FEATURE_IS_UNKNOWN(o, feature_index_y)) ){
		    
		    xc = Plotting.MAP_X_TO_IMAGE(Plotting.GET_FEATURE_VALUE(geot.myDS, o, feature_index_x), width, x_min, x_max);
		    yc = Plotting.MAP_Y_TO_IMAGE(Plotting.GET_FEATURE_VALUE(geot.myDS, o, feature_index_y), height, y_min, y_max);

		    graph.fill(new Ellipse2D.Double(xc-((double) (DATA_SIZE/2)), yc-((double) (DATA_SIZE/2)), (double) DATA_SIZE, (double) DATA_SIZE));
		}
	    }
	    graph.setPaint(Color.black);  	
	}

	
	try{
	    ImageIO.write(img, "PNG", new File(name_file));
	}catch(IOException e){
	}
    }

    public void compute_and_store_x_y_densities(GenerativeModelBasedOnEnsembleOfTrees geot, String name_file){
	// stores the joint density in (x,y) and => image
	int i, j, k, l;
	double xc, yc;
	double xmic, xmac, ymic, ymac;
	
	if ( (Feature.IS_NOMINAL(geot.myDS.features.elementAt(feature_index_x).type)) || (Feature.IS_NOMINAL(geot.myDS.features.elementAt(feature_index_y).type)) )
	    Dataset.perror("Plotting.class :: plotting non continuous / integer variables");

	if (length_x < length_y)
	    WeightedRectangle2D.NUMERICAL_PRECISION_ERROR = length_x / 10000.0;
	else
	    WeightedRectangle2D.NUMERICAL_PRECISION_ERROR = length_y / 10000.0;

	compute_geometric_objects(geot, Plotting.RECTANGLES);

	Vector <WeightedRectangle2D> all_splitted_rectangles = new Vector<>();
	Vector <WeightedRectangle2D> all_splitted_current_rectangles, all_splitted_local_current_rectangles, next_splitted_current_rectangles;

	WeightedRectangle2D duma, dumb;

	all_splitted_rectangles = new Vector<>();
	
	for (i=0;i<all_rectangles.size();i++){	    
	    all_splitted_current_rectangles = new Vector <>();
	    all_splitted_current_rectangles.addElement(all_rectangles.elementAt(i));
	    for (j=0;j<all_rectangles.size();j++){
		if (j!=i){
		    next_splitted_current_rectangles = new Vector <>();
		    for (k=0;k<all_splitted_current_rectangles.size();k++){
			duma = all_splitted_current_rectangles.elementAt(k);
			dumb = all_rectangles.elementAt(j);
			all_splitted_local_current_rectangles = SPLIT_A_FROM_B(duma, dumb);
			next_splitted_current_rectangles.addAll(all_splitted_local_current_rectangles);
		    }
		    all_splitted_current_rectangles = next_splitted_current_rectangles;
		}
	    }

	    all_splitted_rectangles.addAll(all_splitted_current_rectangles);
	}		    

	i = 0;
	do{
	    duma = all_splitted_rectangles.elementAt(i);
	    j = all_splitted_rectangles.size()-1;
	    do{
		dumb = all_splitted_rectangles.elementAt(j);

		if (duma.matches(dumb)){
		    duma.weight += dumb.weight;
		    all_splitted_rectangles.removeElementAt(j);
		}
		j--;
	    }while(j>i);

	    i++;
	}while(i<all_splitted_rectangles.size()-1);

	// CHECK
	boolean infinite_density = false;
	for (i=0;i<all_splitted_rectangles.size()-1;i++){
	    duma = all_splitted_rectangles.elementAt(i);
	    if (duma.surface() == 0.0){
		infinite_density = true;
		Dataset.perror(" INFINITE DENSITY ");
	    }
		
	    for (j=i+1;j<all_splitted_rectangles.size();j++){
		dumb = all_splitted_rectangles.elementAt(j);

		if (duma.matches(dumb))
		    Dataset.perror("MATCH ERROR " + duma + " & " + dumb);
	    }
	}
	
	//computing densities
	
	double total_weight = 0.0, dens;
	for (i=0;i<all_splitted_rectangles.size();i++)
	    total_weight += all_splitted_rectangles.elementAt(i).weight;

	for (i=0;i<all_splitted_rectangles.size();i++){
	    duma = all_splitted_rectangles.elementAt(i);	    
	    dens = (duma.weight / total_weight) / duma.surface();
	    duma.density = dens;
	}
	
	Vector <WeightedRectangle2D> all_final_rectangles = new Vector<>();
	for (i=0;i<all_splitted_rectangles.size();i++){
	    duma = all_splitted_rectangles.elementAt(i);

	    xmic = Plotting.MAP_X_TO_IMAGE(duma.rectangle.x, width, x_min, x_max);
	    xmac = Plotting.MAP_X_TO_IMAGE(duma.rectangle.x + duma.rectangle.width, width, x_min, x_max);
	    ymic = Plotting.MAP_Y_TO_IMAGE(duma.rectangle.y + duma.rectangle.height, height, y_min, y_max);
	    ymac = Plotting.MAP_Y_TO_IMAGE(duma.rectangle.y, height, y_min, y_max);

	    all_final_rectangles.addElement(new WeightedRectangle2D(new Rectangle2D.Double(xmic - OFFSET, ymic - OFFSET, xmac - xmic + (2*OFFSET), ymac - ymic + (2*OFFSET)), duma.weight, duma.density));

	}

	double max_dens = -1.0;
	for (i=0;i<all_final_rectangles.size();i++){
	    duma = all_final_rectangles.elementAt(i);
	    if ( (i==0) || (duma.density > max_dens) )
		max_dens = duma.density;
	}

	BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
	Graphics2D graph = (Graphics2D) img.getGraphics();

	graph.setPaint(HEATMAP_COLOR(0.0f));
	graph.fill(new Rectangle2D.Double(0,0,width,height));
	graph.setPaint(Color.black);
          
        graph.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
	Color fc;
	float cd, enh;
	for (i=0;i<all_final_rectangles.size();i++){
	    duma = all_final_rectangles.elementAt(i);
	    cd = (float) (duma.density / max_dens);		
	    fc = HEATMAP_COLOR(cd);
	    graph.setPaint(fc);
	    graph.fill(duma.rectangle);
	    graph.setPaint(Color.red);
	}
	try{
	    ImageIO.write(img, "PNG", new File(name_file));
	}catch(IOException e){
	}
    }

    public void compute_and_store_x_y_densities_dataset(GenerativeModelBasedOnEnsembleOfTrees geot, Vector <Observation> observations, String name_file){

	Vector <WeightedRectangle2D> all_domain_rectangles = new Vector<>();
	Vector <WeightedRectangle2D> rectangles_containing_o;
	int i, su, j, k;
	Rectangle2D.Double rd;
	Observation o;
	WeightedRectangle2D duma;
    
	boolean unknown_x, unknown_y, in_x, in_y;
	double total_weight = 0.0, dens;
	double xmic, xmac, ymic, ymac;
	double delta_x = length_x / ((double) GRID_SIZE), delta_y = length_y / ((double) GRID_SIZE);

	if (length_x < length_y)
	    WeightedRectangle2D.NUMERICAL_PRECISION_ERROR = length_x / 10000.0;
	else
	    WeightedRectangle2D.NUMERICAL_PRECISION_ERROR = length_y / 10000.0;
	
	for (i=0;i<GRID_SIZE;i++)
	    for (j=0;j<GRID_SIZE;j++)
		all_domain_rectangles.addElement(new WeightedRectangle2D(new Rectangle2D.Double((x_min + ((double) i)*delta_x), y_min + (((double) j)*delta_y), delta_x, delta_y), 0.0));

	for (i=0;i<observations.size();i++){
	    total_weight += 1.0;
	    o = observations.elementAt(i);
	    rectangles_containing_o = new Vector<>();
	    unknown_x = Observation.FEATURE_IS_UNKNOWN(o, feature_index_x);
	    unknown_y = Observation.FEATURE_IS_UNKNOWN(o, feature_index_y);

	    for (j=0;j<all_domain_rectangles.size();j++){
		in_x = ( (unknown_x) || (all_domain_rectangles.elementAt(j).contains_X(GET_FEATURE_VALUE(geot.myDS, o, feature_index_x))) );
		in_y = ( (unknown_y) || (all_domain_rectangles.elementAt(j).contains_Y(GET_FEATURE_VALUE(geot.myDS, o, feature_index_y))) );
		if ( (in_x) && (in_y) )
		    rectangles_containing_o.addElement(all_domain_rectangles.elementAt(j));
	    }

	    if (rectangles_containing_o.size() == 0)
		System.out.print("X");

	    for (j=0;j<rectangles_containing_o.size();j++)
		rectangles_containing_o.elementAt(j).weight += (1.0 / ((double) rectangles_containing_o.size()) );
	}

	for (i=0;i<all_domain_rectangles.size();i++){
	    duma = all_domain_rectangles.elementAt(i);
	    dens = (duma.weight / total_weight) / duma.surface();
	    duma.density = dens;
	}
	    
	Vector <WeightedRectangle2D> all_final_rectangles = new Vector<>();
	for (i=0;i<all_domain_rectangles.size();i++){
	    duma = all_domain_rectangles.elementAt(i);

	    xmic = Plotting.MAP_X_TO_IMAGE(duma.rectangle.x, width, x_min, x_max);
	    xmac = Plotting.MAP_X_TO_IMAGE(duma.rectangle.x + duma.rectangle.width, width, x_min, x_max);
	    ymic = Plotting.MAP_Y_TO_IMAGE(duma.rectangle.y + duma.rectangle.height, height, y_min, y_max);
	    ymac = Plotting.MAP_Y_TO_IMAGE(duma.rectangle.y, height, y_min, y_max);

	    all_final_rectangles.addElement(new WeightedRectangle2D(new Rectangle2D.Double(xmic - OFFSET, ymic - OFFSET, xmac - xmic + (2*OFFSET), ymac - ymic + (2*OFFSET)), duma.weight, duma.density));

	}

	double max_dens = -1.0;
	for (i=0;i<all_final_rectangles.size();i++){
	    duma = all_final_rectangles.elementAt(i);
	    if ( (i==0) || (duma.density > max_dens) )
		max_dens = duma.density;
	}

	BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
	Graphics2D graph = (Graphics2D) img.getGraphics();

	graph.setPaint(Color.white);
	graph.fill(new Rectangle2D.Double(0,0,width,height));
	graph.setPaint(Color.black);
          
        graph.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
	//graph.setStroke(new BasicStroke(2));
	Color fc;
	float cd, enh;
	for (i=0;i<all_final_rectangles.size();i++){
	    duma = all_final_rectangles.elementAt(i);
	    cd = (float) (duma.density / max_dens);

	    fc = HEATMAP_COLOR(cd);
	    graph.setPaint(fc);
	    graph.fill(duma.rectangle);
	    graph.setPaint(Color.red);
	}
	
	try{
	    ImageIO.write(img, "PNG", new File(name_file));
	}catch(IOException e){
	}
    }

    public static Color HEATMAP_COLOR(float f){
	float [] tvals = {0.0f, 0.05f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};
	Color [] cvals = {Color.black, Color.blue, Color.cyan, Color.green, Color.yellow, Color.red, Color.white};

	int i=0;
	boolean found = false;
	do{
	    if ( ( (f>=tvals[i]) && (f<tvals[i+1]) ) || (i == tvals.length-2) )
		found = true;
	    else
		i++;		
	}while(!found);

	float [] c1 = cvals[i].getComponents(null);
	float [] c2 = cvals[i+1].getComponents(null);

	float alpha = (tvals[i+1] - f)/(tvals[i+1] - tvals[i]);

	Color cret = new Color(alpha * c1[0] + (1.0f - alpha) * c2[0]
			       , alpha * c1[1] + (1.0f - alpha) * c2[1]
			       , alpha * c1[2] + (1.0f - alpha) * c2[2]
			       , alpha * c1[3] + (1.0f - alpha) * c2[3]);

	return cret;
    }

    public static Vector<String> WHICH_VERTICES_OF_B_ARE_IN_A(WeightedRectangle2D a, WeightedRectangle2D b){
	Vector <String> all = new Vector <>();
	if (a.strictly_contains(b.rectangle.x, b.rectangle.y))
	    all.addElement(LOWER_LEFT);
	if (a.strictly_contains(b.rectangle.x + b.rectangle.width, b.rectangle.y))
	    all.addElement(LOWER_RIGHT);
	if (a.strictly_contains(b.rectangle.x + b.rectangle.width, b.rectangle.y + b.rectangle.height))
	    all.addElement(UPPER_RIGHT);
	if (a.strictly_contains(b.rectangle.x, b.rectangle.y + b.rectangle.height))
	    all.addElement(UPPER_LEFT);
	return all;
    }
    
    public static Vector<WeightedRectangle2D> SPLIT_A_FROM_B(WeightedRectangle2D a, WeightedRectangle2D b){
	Vector<WeightedRectangle2D> ret = new Vector<>();

	Vector <String> all = WHICH_VERTICES_OF_B_ARE_IN_A(a, b);

	double x, y, w, h;
	WeightedRectangle2D c;
	
	if (all.size() == 4){
	    // b included in a: 9 new rectangles

	    // 1
	    x = a.rectangle.x;
	    y = a.rectangle.y;
	    w = b.rectangle.x - a.rectangle.x;
	    h = b.rectangle.y - a.rectangle.y;
	    c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
	    c.weight = (c.surface() / a.surface()) * a.weight;
	    ret.addElement(c);

	    // 2
	    x = b.rectangle.x;
	    y = a.rectangle.y;
	    w = b.rectangle.width;
	    h = b.rectangle.y - a.rectangle.y;
	    c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
	    c.weight = (c.surface() / a.surface()) * a.weight;
	    ret.addElement(c);

	    // 3
	    x = b.rectangle.x + b.rectangle.width;
	    y = a.rectangle.y;
	    w = a.rectangle.x + a.rectangle.width - (b.rectangle.x + b.rectangle.width);
	    h = b.rectangle.y - a.rectangle.y;
	    c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
	    c.weight = (c.surface() / a.surface()) * a.weight;
	    ret.addElement(c);

	    // 4
	    x = a.rectangle.x;
	    y = b.rectangle.y;
	    w = b.rectangle.x - a.rectangle.x;
	    h = b.rectangle.height;
	    c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
	    c.weight = (c.surface() / a.surface()) * a.weight;
	    ret.addElement(c);

	    // 5
	    x = b.rectangle.x;
	    y = b.rectangle.y;
	    w = b.rectangle.width;
	    h = b.rectangle.height;
	    c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
	    c.weight = (c.surface() / a.surface()) * a.weight;
	    ret.addElement(c);

	    // 6
	    x = b.rectangle.x + b.rectangle.width;
	    y = b.rectangle.y;
	    w = a.rectangle.x + a.rectangle.width - (b.rectangle.x + b.rectangle.width);
	    h = b.rectangle.height;
	    c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
	    c.weight = (c.surface() / a.surface()) * a.weight;
	    ret.addElement(c);

	    // 7
	    x = a.rectangle.x;
	    y = b.rectangle.y + b.rectangle.height;
	    w = b.rectangle.x - a.rectangle.x;
	    h = a.rectangle.y + a.rectangle.height - (b.rectangle.y + b.rectangle.height);
	    c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
	    c.weight = (c.surface() / a.surface()) * a.weight;
	    ret.addElement(c);

	    // 8
	    x = b.rectangle.x;
	    y = b.rectangle.y + b.rectangle.height;
	    w = b.rectangle.width;
	    h = a.rectangle.y + a.rectangle.height - (b.rectangle.y + b.rectangle.height);
	    c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
	    c.weight = (c.surface() / a.surface()) * a.weight;
	    ret.addElement(c);

	    // 9
	    x = b.rectangle.x + b.rectangle.width;
	    y = b.rectangle.y + b.rectangle.height;
	    w = a.rectangle.x + a.rectangle.width - (b.rectangle.x + b.rectangle.width);
	    h = a.rectangle.y + a.rectangle.height - (b.rectangle.y + b.rectangle.height);
	    c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
	    c.weight = (c.surface() / a.surface()) * a.weight;
	    ret.addElement(c);
	}else if (all.size() == 2){
	    // 6 rectangles, 4 configurations

	    if ( (all.contains(LOWER_LEFT)) && (all.contains(UPPER_LEFT)) ){
		// A1
		x = a.rectangle.x;
		y = a.rectangle.y;
		w = b.rectangle.x - a.rectangle.x;
		h = b.rectangle.y - a.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// A2
		x = b.rectangle.x;
		y = a.rectangle.y;
		w = a.rectangle.x + a.rectangle.width - b.rectangle.x;
		h = b.rectangle.y - a.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// A3
		x = a.rectangle.x;
		y = b.rectangle.y;
		w = b.rectangle.x - a.rectangle.x;
		h = b.rectangle.height;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// A4
		x = b.rectangle.x;
		y = b.rectangle.y;
		w = a.rectangle.x + a.rectangle.width - b.rectangle.x;
		h = b.rectangle.height;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// A5
		x = a.rectangle.x;
		y = b.rectangle.y + b.rectangle.height;
		w = b.rectangle.x - a.rectangle.x;
		h = a.rectangle.y + a.rectangle.height - (b.rectangle.y + b.rectangle.height);
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// A6
		x = b.rectangle.x;
		y = b.rectangle.y + b.rectangle.height;
		w = a.rectangle.x + a.rectangle.width - b.rectangle.x;
		h = a.rectangle.y + a.rectangle.height - (b.rectangle.y + b.rectangle.height);
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);
	    }else if ( (all.contains(LOWER_LEFT)) && (all.contains(LOWER_RIGHT)) ){
		
		// B1
		x = a.rectangle.x;
		y = a.rectangle.y;
		w = b.rectangle.x - a.rectangle.x;
		h = b.rectangle.y - a.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// B2
		x = b.rectangle.x;
		y = a.rectangle.y;
		w = b.rectangle.width;
		h = b.rectangle.y - a.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// B3
		x = b.rectangle.x + b.rectangle.width;
		y = a.rectangle.y;
		w = a.rectangle.x + a.rectangle.width - (b.rectangle.x + b.rectangle.width);
		h = b.rectangle.y - a.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// B4
		x = a.rectangle.x;
		y = b.rectangle.y;
		w = b.rectangle.x - a.rectangle.x;
		h = a.rectangle.y + a.rectangle.height - b.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// B5
		x = a.rectangle.x;
		y = b.rectangle.y;
		w = b.rectangle.width;
		h = a.rectangle.y + a.rectangle.height - b.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// B6
		x = b.rectangle.x + b.rectangle.width;
		y = b.rectangle.y;
		w = a.rectangle.x + a.rectangle.width - (b.rectangle.x + b.rectangle.width);
		h = a.rectangle.y + a.rectangle.height - b.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);
	    }else if ( (all.contains(LOWER_RIGHT)) && (all.contains(UPPER_RIGHT)) ){

		// C1
		x = a.rectangle.x;
		y = a.rectangle.y;
		w = b.rectangle.x + b.rectangle.width - a.rectangle.x;
		h = b.rectangle.y - a.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// C2
		x = b.rectangle.x + b.rectangle.width;
		y = a.rectangle.y;
		w = a.rectangle.x + a.rectangle.width - (b.rectangle.x + b.rectangle.width);
		h = b.rectangle.y - a.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// C3
		x = a.rectangle.x;
		y = b.rectangle.y;
		w = b.rectangle.x + b.rectangle.width - a.rectangle.x;
		h = b.rectangle.height;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// C4
		x = b.rectangle.x + b.rectangle.width;
		y = b.rectangle.y;
		w = a.rectangle.x + a.rectangle.width - (b.rectangle.x + b.rectangle.width);
		h = b.rectangle.height;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// C5
		x = a.rectangle.x;
		y = b.rectangle.y + b.rectangle.height;
		w = b.rectangle.x + b.rectangle.width - a.rectangle.x;
		h = a.rectangle.y + a.rectangle.height - (b.rectangle.y + b.rectangle.height);
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// C6
		x = b.rectangle.x + b.rectangle.width;
		y = b.rectangle.y + b.rectangle.height;
		w = a.rectangle.x + a.rectangle.width - (b.rectangle.x + b.rectangle.width);
		h = a.rectangle.y + a.rectangle.height - (b.rectangle.y + b.rectangle.height);
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);
	    }else if ( (all.contains(UPPER_LEFT)) && (all.contains(UPPER_RIGHT)) ){

		// D1
		x = a.rectangle.x;
		y = a.rectangle.y;
		w = b.rectangle.x - a.rectangle.x;
		h = b.rectangle.y + b.rectangle.height - a.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// D2
		x = b.rectangle.x;
		y = a.rectangle.y;
		w = b.rectangle.width;
		h = b.rectangle.y + b.rectangle.height - a.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// D3
		x = b.rectangle.x + b.rectangle.width;
		y = a.rectangle.y;
		w = a.rectangle.x + a.rectangle.width - (b.rectangle.x + b.rectangle.width);
		h = b.rectangle.y + b.rectangle.height - a.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// D4
		x = a.rectangle.x;
		y = b.rectangle.y + b.rectangle.height;
		w = b.rectangle.x - a.rectangle.x;
		h = a.rectangle.y + a.rectangle.height - (b.rectangle.y + b.rectangle.height);
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// D5
		x = b.rectangle.x;
		y = b.rectangle.y + b.rectangle.height;
		w = b.rectangle.width;
		h = a.rectangle.y + a.rectangle.height - (b.rectangle.y + b.rectangle.height);
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// D6
		x = b.rectangle.x + b.rectangle.width;
		y = b.rectangle.y + b.rectangle.height;
		w = a.rectangle.x + a.rectangle.width - (b.rectangle.x + b.rectangle.width);
		h = a.rectangle.y + a.rectangle.height - (b.rectangle.y + b.rectangle.height);
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);		
	    }else
		Dataset.perror("Plotting.class :: no such 2 configuration");
	}else if (all.size() == 1){
	    // 4 rectangles, 4 configurations

	    if (all.contains(LOWER_LEFT)){
		// A1
		x = a.rectangle.x;
		y = a.rectangle.y;
		w = b.rectangle.x - a.rectangle.x;
		h = b.rectangle.y - a.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// A2
		x = b.rectangle.x;
		y = a.rectangle.y;
		w = a.rectangle.x + a.rectangle.width - b.rectangle.x;
		h = b.rectangle.y - a.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// A3
		x = a.rectangle.x;
		y = b.rectangle.y;
		w = b.rectangle.x - a.rectangle.x;
		h = a.rectangle.y + a.rectangle.height - b.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// A4
		x = b.rectangle.x;
		y = b.rectangle.y;
		w = a.rectangle.x + a.rectangle.width - b.rectangle.x;
		h = a.rectangle.y + a.rectangle.height - b.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);
	    } else if (all.contains(LOWER_RIGHT)){
		// B1
		x = a.rectangle.x;
		y = a.rectangle.y;
		w = b.rectangle.x + b.rectangle.width - a.rectangle.x;
		h = b.rectangle.y - a.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// B2
		x = b.rectangle.x + b.rectangle.width ;
		y = a.rectangle.y;
		w = a.rectangle.x + a.rectangle.width - (b.rectangle.x + b.rectangle.width);
		h = b.rectangle.y - a.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// B3
		x = a.rectangle.x;
		y = b.rectangle.y;
		w = b.rectangle.x + b.rectangle.width - a.rectangle.x;
		h = a.rectangle.y + a.rectangle.height - b.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// B4
		x = b.rectangle.x + b.rectangle.width ;
		y = b.rectangle.y;
		w = a.rectangle.x + a.rectangle.width - (b.rectangle.x + b.rectangle.width);
		h = a.rectangle.y + a.rectangle.height - b.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);
	    } else if (all.contains(UPPER_RIGHT)){
		// C1
		x = a.rectangle.x;
		y = a.rectangle.y;
		w = b.rectangle.x + b.rectangle.width - a.rectangle.x;
		h = b.rectangle.y + b.rectangle.height - a.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// C2
		x = b.rectangle.x + b.rectangle.width;
		y = a.rectangle.y;
		w = a.rectangle.x + a.rectangle.width - (b.rectangle.x + b.rectangle.width);
		h = b.rectangle.y + b.rectangle.height - a.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// C3
		x = a.rectangle.x;
		y = b.rectangle.y + b.rectangle.height;
		w = b.rectangle.x + b.rectangle.width - a.rectangle.x;
		h = a.rectangle.y + a.rectangle.height - (b.rectangle.y + b.rectangle.height);
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// C4
		x = b.rectangle.x + b.rectangle.width;
		y = b.rectangle.y + b.rectangle.height;
		w = a.rectangle.x + a.rectangle.width - (b.rectangle.x + b.rectangle.width);
		h = a.rectangle.y + a.rectangle.height - (b.rectangle.y + b.rectangle.height);
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);
	    } else if (all.contains(UPPER_LEFT)){
		// D1
		x = a.rectangle.x;
		y = a.rectangle.y;
		w = b.rectangle.x - a.rectangle.x;
		h = b.rectangle.y + b.rectangle.height - a.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// D2
		x = b.rectangle.x;
		y = a.rectangle.y;
		w = a.rectangle.x + a.rectangle.width - b.rectangle.x;
		h = b.rectangle.y + b.rectangle.height - a.rectangle.y;
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// D3
		x = a.rectangle.x;
		y = b.rectangle.y + b.rectangle.height;
		w = b.rectangle.x - a.rectangle.x;
		h = a.rectangle.y + a.rectangle.height - (b.rectangle.y + b.rectangle.height);
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);

		// D4
		x = b.rectangle.x;
		y = b.rectangle.y + b.rectangle.height;
		w = a.rectangle.x + a.rectangle.width - b.rectangle.x;
		h = a.rectangle.y + a.rectangle.height - (b.rectangle.y + b.rectangle.height);
		c = new WeightedRectangle2D(new Rectangle2D.Double(x, y, w, h), 0.0);
		c.weight = (c.surface() / a.surface()) * a.weight;
		ret.addElement(c);
	    }else
		Dataset.perror("Plotting.class :: no such 1 configuration");
	}else if (all.size() == 0){
	    // no intersection
	    ret.addElement(a);
	}else
	    Dataset.perror("Plotting.class :: no such all configuration");
	
	return ret;
    }
    
}
