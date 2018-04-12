package clustering;

import java.io.BufferedReader;
import java.io.FileReader;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.bayes.NaiveBayesMultinomialUpdateable;
import weka.classifiers.bayes.NaiveBayesSimple;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.SMOset;
import weka.classifiers.mi.supportVector.MIPolyKernel;
import weka.classifiers.pmml.consumer.NeuralNetwork;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.PruneableClassifierTree;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;


public class classifierTest {
	public static void main(String[] args) throws Exception
	{
		/////for the training file
		BufferedReader br = new BufferedReader(new FileReader("TrainOnly.arff"));
			Instances train = new Instances(br);
		
			train.setClass(train.attribute("s_class"));

			StringToWordVector v = new StringToWordVector();
			v.setOutputWordCounts(true);
//			v.setUseStoplist(false);
			v.setInputFormat(train);
			train = Filter.useFilter(train, v);
			/////////////////// for the test file
			BufferedReader br2 = new BufferedReader(new FileReader("Test.arff"));
			Instances test = new Instances(br2);
		
			test.setClass(test.attribute("s_class"));

			//StringToWordVector v2 = new StringToWordVector();
			v.setOutputWordCounts(true);
//			v2.setUseStoplist(false);
			v.setInputFormat(test);
			test = Filter.useFilter(test, v);

		weka.classifiers.bayes.NaiveBayesMultinomial c = new NaiveBayesMultinomial();//original
		//RandomForest c=new RandomForest();
		//c.setNumTrees(9);//for random forest only
		//	weka.classifiers.bayes.NaiveBayesUpdateable c = new NaiveBayesUpdateable();
			//weka.classifiers.functions.SMO c = new weka.classifiers.functions.SMO();
			//weka.classifiers.trees.j48.PruneableClassifierTree c2 = new PruneableClassifierTree(toSelectLocModel, pruneTree, num, cleanup, seed);//= new J48();
		//weka.classifiers.mi.supportVector.MIPolyKernel c= new MIPolyKernel();
		c.buildClassifier(train);
		for(int i=0;i<test.numInstances();i++) {
		//System.out.println(i);
		System.out.println(test.attribute("s_class").value((int)c.classifyInstance(test.instance(i))));}// +" ------> "+ test.attribute("s_class").value((int)test.instance(i).classValue()));
		/*double pred = c.classifyInstance(test.instance(i));
		System.out.print("ID: " + test.instance(i).value(0));
		System.out.print(", actual: " + test.classAttribute().value((int) test.instance(i).classValue()));
		System.out.println(", predicted: " + test.classAttribute().value((int) pred));}*/
	}

}
