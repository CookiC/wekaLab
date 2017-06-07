package MOFES;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.DoubleBuffer;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;
import java.util.prefs.PreferenceChangeEvent;

import javax.print.attribute.standard.MediaName;
import javax.swing.text.html.HTML.Tag;

import expansion.Recorder;
import expansion.dataIn;
import expansion.dataOut;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.GeneticSearch;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.NSGAII;
import weka.attributeSelection.NSGAIIP;
import weka.attributeSelection.NSGAIIS;
import weka.attributeSelection.RegressionEval;
import weka.attributeSelection.SingleWrapperSubsetEval;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.AbstractEvaluationMetric;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.Instances;
import weka.gui.knowledgeflow.steps.ImageViewerInteractiveView;

public class MOFESP extends expansion.experiment{
	private final String dataSetsName="NASA";
	
	private final String[] searchNames={"MOFESP","MOFESS"};
	
	private final String[] statNames={"FN","AUC"};
	
	private final int runtimes=1;
	
	private final Random rnd=new Random(13);
	
	private NSGAII[] searchs=new NSGAII[2];
	
	private ArrayList<Attribute> resAttrs=new ArrayList<Attribute>();
	
	private double[][] time;
	
	public MOFESP() throws FileNotFoundException, IOException{
		input=new dataIn(dataSetsName);
		output=new dataOut(dataSetsName);
		dataSet=input.getDataSet(0);
		time=new double[input.size()][2];
		
		NSGAIIS nsgaiis=new NSGAIIS();
		NSGAIIP nsgaiip=new NSGAIIP();
		nsgaiip.setThreadsNum(Runtime.getRuntime().availableProcessors());
		searchs[0]=nsgaiip;
		searchs[1]=nsgaiis;
		for(NSGAII m:searchs){
			m.setPopulationSize(40);
			m.setMaxGenerations(30);
			m.setCrossoverProb(0.2);
			m.setMutationProb(0.6);
		}
		for(String m:statNames)
			resAttrs.add(new Attribute(m));
	}
	
	private long runOnce(NSGAII search,int seed) throws Exception {
		long start;
		long time=0;
		IBk classifer=new IBk(7);
		WrapperSubsetEval wrap=new WrapperSubsetEval();
		wrap.setClassifier(classifer);
		wrap.setEvaluationMeasure("auc");
		wrap.setFolds(3);
		wrap.buildEvaluator(trainSet);
		wrap.setSeed(seed);

		start=System.currentTimeMillis();
		search.search(wrap, trainSet,statNames);
		time+=System.currentTimeMillis()-start;
		return time;
	}
	
	public void run() throws Exception{
		int i,j,k;
		String[] rowNames=new String[input.size()];
		for(i=0;i<input.size();++i){
			dataSet=input.getDataSet(i);
			dataSet.setClassIndex("Defective");
			System.out.println(dataSet.relationName());
			rowNames[i]=new String(dataSet.relationName());
			randomSplit(dataSet, 0.7,1);
			for(j=0;j<runtimes;++j){
				int seed=rnd.nextInt();
				for(k=0;k<searchs.length;++k){
					time[i][k]+=runOnce(searchs[k],seed)/1000.0;
				}
			}
		}
		output.setSubDir("");
		output.outCSV("Computational Cost",rowNames,searchNames,time);
	}
	
	public static void main(String[] args)throws Exception{
		MOFESP mofesp=new MOFESP();
		mofesp.run();
	}
}
