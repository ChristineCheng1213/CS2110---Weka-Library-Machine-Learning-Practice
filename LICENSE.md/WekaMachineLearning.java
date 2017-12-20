	import java.util.*;
	import java.io.BufferedReader;
	import java.io.FileNotFoundException;
	import java.io.FileReader;
	import weka.classifiers.Classifier;
	import weka.classifiers.Evaluation;
	import weka.classifiers.evaluation.NominalPrediction;
	import weka.classifiers.rules.PART;
	import weka.classifiers.rules.JRip;
	import weka.classifiers.trees.J48;
	import weka.core.FastVector;
	import weka.core.Instances;
	
	//source used:https://www.programcreek.com/2013/01/a-simple-machine-learning-example-in-java/ and my friend  
	
public class WekaMachineLearning {

	public static Scanner in = new Scanner(System.in);
	
	public static BufferedReader readFile(String filename) {
		BufferedReader input = null;
		
		try {
			input = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found");}
 		return input;}
	
	
	public static Evaluation classify(Classifier model, Instances trainingSet, Instances testingSet) throws Exception {
		Evaluation evaluation = new Evaluation(trainingSet);
		model.buildClassifier(trainingSet);
		evaluation.evaluateModel(model, testingSet);
		return evaluation;
	}
	
	
	public static double calculateAccuracy(FastVector prediction) {
		double ct = 0;
		
		for (int i = 0; i < prediction.size(); i++) {
			NominalPrediction np = (NominalPrediction) prediction.elementAt(i);
			if (np.predicted() == np.actual()) {
				ct++;
				}
			}
		return 100 * ct / prediction.size();
	}
	
	
	public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
		Instances[][] split = new Instances[2][numberOfFolds];
		
		for (int i = 0; i < numberOfFolds; i++) {
			split[0][i] = data.trainCV(numberOfFolds, i);
			split[1][i] = data.testCV(numberOfFolds, i);
		}
		return split;
	}
	
	
	public static int getK() {
		int k = 0;
		do{
		System.out.println("Enter the number of folds (k) split cross validations.");
		System.out.println("Entering '5' will give you 80% Training and 20% Testing.");
		System.out.println("Entering '10' will give you 90% Training and 10% Testing.");
		k = in.nextInt();
		in.nextLine();
		if (k <2) {
			System.out.println("ERROR: Please enter an integer greater than 2.");}
		} while (k<2);
		return k;
	}
	
	
	public static int getAlgo() {
		int k = 1;
		do {
			System.out.println("What kind of algorithm you would like to use?:");
			System.out.println("1. J48");
			System.out.println("2. JRip");
			System.out.println("3. PART");
			k = in.nextInt();
			in.nextLine();
			if (k < 1 || k > 3) {
				System.out.println("Warning!!! Input out of range.");}
		} while (k < 1 || k > 3);
		return k;
	}
	
	
	public static void main(String[] args) throws Exception {
		BufferedReader datafile = readFile("nursery.arff");
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
 		
		int k = getK();
		Instances[][] split = crossValidationSplit(data, k);
 
		Instances[] trainingSet = split[0];
		Instances[] testingSet = split[1];
 
		 Classifier algo;
				int c = getAlgo();
				if (c == 1) {
					algo = new J48();
				} else if (c == 2) {
					algo = new JRip();
				} else {
					algo = new PART();
				}
	
			FastVector predictions = new FastVector();
			for (int i = 0; i < trainingSet.length; i++) {
				Evaluation validation = classify(algo, trainingSet[i], testingSet[i]);
				predictions.appendElements(validation.predictions());
			}
			System.out.println(algo.toString());
			
			double accuracy = calculateAccuracy(predictions);

			System.out.println("Accuracy of " + algo.getClass().getSimpleName() + ": "
				+ String.format("%.2f%%", accuracy) + "\n---------------------------------");
		}

}
