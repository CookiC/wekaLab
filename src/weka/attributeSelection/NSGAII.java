/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    GeneticSearch.java
 *    Copyright (C) 1999 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.attributeSelection;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.BitSet;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Set;
import java.util.TreeSet;
import java.util.Vector;

import weka.attributeSelection.*;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Range;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;

/**
 * <!-- globalinfo-start --> GeneticSearch:<br/>
 * <br/>
 * Performs a search using the simple genetic algorithm described in Goldberg
 * (1989).<br/>
 * <br/>
 * For more information see:<br/>
 * <br/>
 * David E. Goldberg (1989). Genetic algorithms in search, optimization and
 * machine learning. Addison-Wesley.
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <pre>
 * &#64;book{Goldberg1989,
 *    author = {David E. Goldberg},
 *    publisher = {Addison-Wesley},
 *    title = {Genetic algorithms in search, optimization and machine learning},
 *    year = {1989},
 *    ISBN = {0201157675}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -P &lt;start set&gt;
 *  Specify a starting set of attributes.
 *  Eg. 1,3,5-7.If supplied, the starting set becomes
 *  one member of the initial random
 *  population.
 * </pre>
 * 
 * <pre>
 * -Z &lt;population size&gt;
 *  Set the size of the population (even number).
 *  (default = 20).
 * </pre>
 * 
 * <pre>
 * -G &lt;number of generations&gt;
 *  Set the number of generations.
 *  (default = 20)
 * </pre>
 * 
 * <pre>
 * -C &lt;probability of crossover&gt;
 *  Set the probability of crossover.
 *  (default = 0.6)
 * </pre>
 * 
 * <pre>
 * -M &lt;probability of mutation&gt;
 *  Set the probability of mutation.
 *  (default = 0.033)
 * </pre>
 * 
 * <pre>
 * -R &lt;report frequency&gt;
 *  Set frequency of generation reports.
 *  e.g, setting the value to 5 will 
 *  report every 5th generation
 *  (default = number of generations)
 * </pre>
 * 
 * <pre>
 * -S &lt;seed&gt;
 *  Set the random number seed.
 *  (default = 1)
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * @author Mark Hall (mhall@cs.waikato.ac.nz)
 * @version $Revision: 10325 $
 */
public class NSGAII extends ASSearch implements StartSetHandler, OptionHandler, TechnicalInformationHandler {

	/** for serialization */
	static final long serialVersionUID = -1618264232838472679L;

	/**
	 * holds a starting set as an array of attributes. Becomes one member of the
	 * initial random population
	 */
	
	protected boolean debug_print_pop;
	
	protected boolean mode_remove_repetitive_pop;
	
	protected int[] m_starting;

	/** holds the start set for the search as a Range */
	protected Range m_startRange;

	/** does the data have a class */
	protected boolean m_hasClass;

	/** holds the class index */
	protected int m_classIndex;

	/** number of attributes in the data */
	protected int m_numAttribs;
	
	/** the current population */
	protected GABitSet[] m_population;

	/** the number of individual solutions */
	protected int m_popSize;

	/** the number of entries to cache for lookup */
	protected int m_lookupTableSize;
	
	/** the lookup table */
	protected Hashtable<BitSet, GABitSet> m_lookupTable;
	
	/** seed for random number generation */
	protected int m_seed;

	/** the probability of crossover occuring */
	protected double m_pCrossover;

	/** the probability of mutation occuring */
	protected double m_pMutation;

	/** the maximum number of generations to evaluate */
	protected int m_maxGenerations;

	/** how often reports are generated */
	protected int m_reportFrequency;

	/** holds the generation reports */
	protected StringBuffer m_generationReports;

	protected int m_objects;
	
	/** the evaluator of objective */
	protected WrapperSubsetEval[] ASEvaluator;

	/** the evaluator of objective */
	protected String m_stateName[];
	
	// Inner class
	/**
	 * A bitset for the genetic algorithm
	 */

	protected class MyDoub {
		public double[] d;

		public MyDoub() {
			d = new double[m_objects];
		}

		public MyDoub(double n) {
			d = new double[m_objects];
			set(n);
		}

		public MyDoub(MyDoub n) {
			d = new double[m_objects];
			clone(n);
		}

		double get(int n) {
			return d[n];
		}
		
		void set(double n) {
			for (int i=0;i<m_objects;++i)
				d[i] = n;
		}
		
		void set(int i,double n) {
			d[i]=n;
		}

		void clone(MyDoub n) {
			for (int i = 0; i != m_objects; ++i)
				d[i] = n.d[i];
		}

		MyDoub plus(double n) {
			MyDoub tmp = new MyDoub();
			for (int i = 0; i != m_objects; ++i)
				tmp.d[i] = d[i] + n;
			return tmp;
		}

		MyDoub plus(MyDoub n) {
			MyDoub tmp = new MyDoub();
			for (int i = 0; i != m_objects; ++i)
				tmp.d[i] = d[i] + n.d[i];
			return tmp;
		}

		MyDoub minus(double n) {
			MyDoub tmp = new MyDoub();
			for (int i = 0; i != m_objects; ++i)
				tmp.d[i] = d[i] - n;
			return tmp;
		}

		MyDoub minus(MyDoub n) {
			MyDoub tmp = new MyDoub();
			for (int i = 0; i != m_objects; ++i)
				tmp.d[i] = d[i] - n.d[i];
			return tmp;
		}

		MyDoub multiply(double n) {
			MyDoub tmp = new MyDoub();
			for (int i = 0; i != m_objects; ++i)
				tmp.d[i] = d[i] * n;
			return tmp;
		}

		MyDoub multiply(MyDoub n) {
			MyDoub tmp = new MyDoub();
			for (int i = 0; i != m_objects; ++i)
				tmp.d[i] = d[i] * n.d[i];
			return tmp;
		}

		MyDoub divide(double n) {
			MyDoub tmp = new MyDoub();
			for (int i = 0; i != m_objects; ++i)
				tmp.d[i] = d[i] / n;
			return tmp;
		}

		MyDoub divide(MyDoub n) {
			MyDoub tmp = new MyDoub();
			for (int i = 0; i != m_objects; ++i)
				tmp.d[i] = d[i] / n.d[i];
			return tmp;
		}

		boolean biggerThan(MyDoub n) {
			for (int i = 0; i != m_objects; ++i)
				if (d[i] <= n.d[i])
					return false;
			return true;
		}

		boolean smallerThan(MyDoub n) {
			for (int i = 0; i != m_objects; ++i)
				if (d[i] >= n.d[i])
					return false;
			return true;
		}

		boolean equalTo(double n) {
			for (int i = 0; i != m_objects; ++i)
				if (d[i] != n)
					return false;
			return true;
		}

		boolean equalTo(MyDoub n) {
			for (int i = 0; i != m_objects; ++i)
				if (d[i] != n.d[i])
					return false;
			return true;
		}
	}

	protected class GABitSet implements Cloneable, Serializable, RevisionHandler {

		/** for serialization */
		static final long serialVersionUID = -2930607837482622224L;

		/** the bitset */
		private BitSet m_chromosome;

		/** holds raw merit */
		private MyDoub m_objective;

		/** the n (pops that dominate it) by XieNaoban */
		public int n;

		/** the s (pops that are dominated by it) by XieNaoban */
		public int s;
		public GABitSet[] sArr;

		/** the n (pops that dominate it) by XieNaoban */
		public int rank;

		/** the distance (pops that dominate it) by XieNaoban */
		public double d;

		/**
		 * Constructor
		 */
		public GABitSet() {
			m_objective = new MyDoub();
			m_chromosome = new BitSet();
			s = 0;
			n = 0;
			d = 0;
			rank = -1;
			sArr = new GABitSet[m_popSize * 2];
		}

		/**
		 * makes a copy of this GABitSet
		 * 
		 * @return a copy of the object
		 * @throws CloneNotSupportedException
		 *             if something goes wrong
		 */
		@Override
		public GABitSet clone() {
			GABitSet temp = new GABitSet();

			temp.setObjective(this.getObjective());
			temp.setChromosome((BitSet) (this.m_chromosome.clone()));
			temp.s = 0;
			temp.n = 0;
			temp.d = 0;
			temp.rank = rank;
			temp.sArr = new GABitSet[m_popSize * 2];
			return temp;
			// return super.clone();
		}

		/**
		 * sets the objective merit value
		 * 
		 * @param objective
		 *            the objective value of this population member
		 */
		public void setObjective(MyDoub objective) {
			for (int i = 0; i != m_objects; ++i)
				m_objective.d[i] = objective.d[i];
		}

		/**
		 * gets the objective merit
		 * 
		 * @return the objective merit of this population member
		 */
		public MyDoub getObjective() {
			return m_objective;
		}
		
		/**
		 * get the chromosome
		 * 
		 * @return the chromosome of this population member
		 */
		public BitSet getChromosome() {
			return m_chromosome;
		}

		/**
		 * set the chromosome
		 * 
		 * @param c
		 *            the chromosome to be set for this population member
		 */
		public void setChromosome(BitSet c) {
			m_chromosome = c;
		}

		/**
		 * unset a bit in the chromosome
		 * 
		 * @param bit
		 *            the bit to be cleared
		 */
		public void clear(int bit) {
			m_chromosome.clear(bit);
		}

		/**
		 * set a bit in the chromosome
		 * 
		 * @param bit
		 *            the bit to be set
		 */
		public void set(int bit) {
			m_chromosome.set(bit);
		}

		/**
		 * get the value of a bit in the chromosome
		 * 
		 * @param bit
		 *            the bit to query
		 * @return the value of the bit
		 */
		public boolean get(int bit) {
			return m_chromosome.get(bit);
		}

		/**
		 * Returns the revision string.
		 * 
		 * @return the revision
		 */
		@Override
		public String getRevision() {
			return RevisionUtils.extract("$Revision: 10325 $");
		}

	}

	/**
	 * Returns an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 **/
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>(7);

		newVector.addElement(new Option("\tSpecify a starting set of attributes." + "\n\tEg. 1,3,5-7."
				+ "If supplied, the starting set becomes" + "\n\tone member of the initial random" + "\n\tpopulation.",
				"P", 1, "-P <start set>"));
		newVector.addElement(new Option("\tSet the size of the population (even number)." + "\n\t(default = 20).", "Z",
				1, "-Z <population size>"));
		newVector.addElement(new Option("\tSet the number of generations." + "\n\t(default = 20)", "G", 1,
				"-G <number of generations>"));
		newVector.addElement(new Option("\tSet the probability of crossover." + "\n\t(default = 0.6)", "C", 1,
				"-C <probability of" + " crossover>"));
		newVector.addElement(new Option("\tSet the probability of mutation." + "\n\t(default = 0.033)", "M", 1,
				"-M <probability of mutation>"));

		newVector.addElement(new Option(
				"\tSet frequency of generation reports." + "\n\te.g, setting the value to 5 will "
						+ "\n\treport every 5th generation" + "\n\t(default = number of generations)",
				"R", 1, "-R <report frequency>"));
		newVector.addElement(new Option("\tSet the random number seed." + "\n\t(default = 1)", "S", 1, "-S <seed>"));
		return newVector.elements();
	}

	/**
	 * Parses a given list of options.
	 * <p/>
	 * 
	 * <!-- options-start --> Valid options are:
	 * <p/>
	 * 
	 * <pre>
	 * -P &lt;start set&gt;
	 *  Specify a starting set of attributes.
	 *  Eg. 1,3,5-7.If supplied, the starting set becomes
	 *  one member of the initial random
	 *  population.
	 * </pre>
	 * 
	 * <pre>
	 * -Z &lt;population size&gt;
	 *  Set the size of the population (even number).
	 *  (default = 20).
	 * </pre>
	 * 
	 * <pre>
	 * -G &lt;number of generations&gt;
	 *  Set the number of generations.
	 *  (default = 20)
	 * </pre>
	 * 
	 * <pre>
	 * -C &lt;probability of crossover&gt;
	 *  Set the probability of crossover.
	 *  (default = 0.6)
	 * </pre>
	 * 
	 * <pre>
	 * -M &lt;probability of mutation&gt;
	 *  Set the probability of mutation.
	 *  (default = 0.033)
	 * </pre>
	 * 
	 * <pre>
	 * -R &lt;report frequency&gt;
	 *  Set frequency of generation reports.
	 *  e.g, setting the value to 5 will 
	 *  report every 5th generation
	 *  (default = number of generations)
	 * </pre>
	 * 
	 * <pre>
	 * -S &lt;seed&gt;
	 *  Set the random number seed.
	 *  (default = 1)
	 * </pre>
	 * 
	 * <!-- options-end -->
	 * 
	 * @param options
	 *            the list of options as an array of strings
	 * @throws Exception
	 *             if an option is not supported
	 * 
	 **/
	@Override
	public void setOptions(String[] options) throws Exception {
		String optionString;
		resetOptions();

		optionString = Utils.getOption('P', options);
		if (optionString.length() != 0) {
			setStartSet(optionString);
		}

		optionString = Utils.getOption('Z', options);
		if (optionString.length() != 0) {
			setPopulationSize(Integer.parseInt(optionString));
		}

		optionString = Utils.getOption('G', options);
		if (optionString.length() != 0) {
			setMaxGenerations(Integer.parseInt(optionString));
			setReportFrequency(Integer.parseInt(optionString));
		}

		optionString = Utils.getOption('C', options);
		if (optionString.length() != 0) {
			setCrossoverProb((new Double(optionString)).doubleValue());
		}

		optionString = Utils.getOption('M', options);
		if (optionString.length() != 0) {
			setMutationProb((new Double(optionString)).doubleValue());
		}

		optionString = Utils.getOption('R', options);
		if (optionString.length() != 0) {
			setReportFrequency(Integer.parseInt(optionString));
		}

		optionString = Utils.getOption('S', options);
		if (optionString.length() != 0) {
			setSeed(Integer.parseInt(optionString));
		}

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Gets the current settings of ReliefFAttributeEval.
	 * 
	 * @return an array of strings suitable for passing to setOptions()
	 */
	@Override
	public String[] getOptions() {

		Vector<String> options = new Vector<String>();

		if (!(getStartSet().equals(""))) {
			options.add("-P");
			options.add("" + startSetToString());
		}
		options.add("-Z");
		options.add("" + getPopulationSize());
		options.add("-G");
		options.add("" + getMaxGenerations());
		options.add("-C");
		options.add("" + getCrossoverProb());
		options.add("-M");
		options.add("" + getMutationProb());
		options.add("-R");
		options.add("" + getReportFrequency());
		options.add("-S");
		options.add("" + getSeed());

		return options.toArray(new String[0]);
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String startSetTipText() {
		return "Set a start point for the search. This is specified as a comma "
				+ "seperated list off attribute indexes starting at 1. It can include "
				+ "ranges. Eg. 1,2,5-9,17. The start set becomes one of the population "
				+ "members of the initial population.";
	}

	/**
	 * Sets a starting set of attributes for the search. It is the search
	 * method's responsibility to report this start set (if any) in its
	 * toString() method.
	 * 
	 * @param startSet
	 *            a string containing a list of attributes (and or ranges), eg.
	 *            1,2,6,10-15.
	 * @throws Exception
	 *             if start set can't be set.
	 */
	@Override
	public void setStartSet(String startSet) throws Exception {
		m_startRange.setRanges(startSet);
	}

	/**
	 * Returns a list of attributes (and or attribute ranges) as a String
	 * 
	 * @return a list of attributes (and or attribute ranges)
	 */
	@Override
	public String getStartSet() {
		return m_startRange.getRanges();
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String seedTipText() {
		return "Set the random seed.";
	}

	/**
	 * set the seed for random number generation
	 * 
	 * @param s
	 *            seed value
	 */
	public void setSeed(int s) {
		m_seed = s;
	}

	/**
	 * get the value of the random number generator's seed
	 * 
	 * @return the seed for random number generation
	 */
	public int getSeed() {
		return m_seed;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String reportFrequencyTipText() {
		return "Set how frequently reports are generated. Default is equal to "
				+ "the number of generations meaning that a report will be printed for "
				+ "initial and final generations. Setting the value to 5 will result in "
				+ "a report being printed every 5 generations.";
	}

	/**
	 * set how often reports are generated
	 * 
	 * @param f
	 *            generate reports every f generations
	 */
	public void setReportFrequency(int f) {
		m_reportFrequency = f;
	}

	/**
	 * get how often repports are generated
	 * 
	 * @return how often reports are generated
	 */
	public int getReportFrequency() {
		return m_reportFrequency;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String mutationProbTipText() {
		return "Set the probability of mutation occuring.";
	}

	/**
	 * set the probability of mutation
	 * 
	 * @param m
	 *            the probability for mutation occuring
	 */
	public void setMutationProb(double m) {
		m_pMutation = m;
	}

	/**
	 * get the probability of mutation
	 * 
	 * @return the probability of mutation occuring
	 */
	public double getMutationProb() {
		return m_pMutation;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String crossoverProbTipText() {
		return "Set the probability of crossover. This is the probability that "
				+ "two population members will exchange genetic material.";
	}

	/**
	 * set the probability of crossover
	 * 
	 * @param c
	 *            the probability that two population members will exchange
	 *            genetic material
	 */
	public void setCrossoverProb(double c) {
		m_pCrossover = c;
	}

	/**
	 * get the probability of crossover
	 * 
	 * @return the probability of crossover
	 */
	public double getCrossoverProb() {
		return m_pCrossover;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String maxGenerationsTipText() {
		return "Set the number of generations to evaluate.";
	}

	/**
	 * set the number of generations to evaluate
	 * 
	 * @param m
	 *            the number of generations
	 */
	public void setMaxGenerations(int m) {
		m_maxGenerations = m;
	}

	/**
	 * get the number of generations
	 * 
	 * @return the maximum number of generations
	 */
	public int getMaxGenerations() {
		return m_maxGenerations;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String populationSizeTipText() {
		return "Set the population size (even number), this is the number of individuals "
				+ "(attribute sets) in the population.";
	}

	/**
	 * set the population size
	 * 
	 * @param p
	 *            the size of the population
	 */
	public void setPopulationSize(int p) {
		if (p % 2 == 0) {
			m_popSize = p;
		} else {
			System.err.println("Population size needs to be an even number!");
		}
	}

	/**
	 * get the size of the population
	 * 
	 * @return the population size
	 */
	public int getPopulationSize() {
		return m_popSize;
	}

	/**
	 * Returns a string describing this search method
	 * 
	 * @return a description of the search suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {
		return "GeneticSearch:\n\nPerforms a search using the simple genetic "
				+ "algorithm described in Goldberg (1989).\n\n" + "For more information see:\n\n"
				+ getTechnicalInformation().toString();
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing detailed
	 * information about the technical background of this class, e.g., paper
	 * reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;

		result = new TechnicalInformation(Type.BOOK);
		result.setValue(Field.AUTHOR, "David E. Goldberg");
		result.setValue(Field.YEAR, "1989");
		result.setValue(Field.TITLE, "Genetic algorithms in search, optimization and machine learning");
		result.setValue(Field.ISBN, "0201157675");
		result.setValue(Field.PUBLISHER, "Addison-Wesley");

		return result;
	}

	/**
	 * Constructor. Make a new GeneticSearch object
	 */
	public NSGAII() {
		resetOptions();
	}

	/**
	 * converts the array of starting attributes to a string. This is used by
	 * getOptions to return the actual attributes specified as the starting set.
	 * This is better than using m_startRanges.getRanges() as the same start set
	 * can be specified in different ways from the command line---eg 1,2,3 ==
	 * 1-3. This is to ensure that stuff that is stored in a database is
	 * comparable.
	 * 
	 * @return a comma seperated list of individual attribute numbers as a
	 *         String
	 */
	private String startSetToString() {
		StringBuffer FString = new StringBuffer();
		boolean didPrint;

		if (m_starting == null) {
			return getStartSet();
		}

		for (int i = 0; i < m_starting.length; i++) {
			didPrint = false;

			if ((m_hasClass == false) || (m_hasClass == true && i != m_classIndex)) {
				FString.append((m_starting[i] + 1));
				didPrint = true;
			}

			if (i == (m_starting.length - 1)) {
				FString.append("");
			} else {
				if (didPrint) {
					FString.append(",");
				}
			}
		}

		return FString.toString();
	}

	/**
	 * returns a description of the search
	 * 
	 * @return a description of the search as a String
	 */
	@Override
	public String toString() {
		StringBuffer GAString = new StringBuffer();
		GAString.append("\tGenetic search.\n\tStart set: ");

		if (m_starting == null) {
			GAString.append("no attributes\n");
		} else {
			GAString.append(startSetToString() + "\n");
		}
		GAString.append("\tPopulation size: " + m_popSize);
		GAString.append("\n\tNumber of generations: " + m_maxGenerations);
		GAString.append("\n\tProbability of crossover: " + Utils.doubleToString(m_pCrossover, 6, 3));
		GAString.append("\n\tProbability of mutation: " + Utils.doubleToString(m_pMutation, 6, 3));
		GAString.append("\n\tReport frequency: " + m_reportFrequency);
		GAString.append("\n\tRandom number seed: " + m_seed + "\n");
		GAString.append(m_generationReports.toString());
		return GAString.toString();
	}

	/**
	 * converts a BitSet into a list of attribute indexes
	 * 
	 * @param group
	 *            the BitSet to convert
	 * @return an array of attribute indexes
	 **/
	protected int[] attributeList(BitSet group) {
		int count = 0;
		// count how many were selected
		for (int i = 0; i < m_numAttribs; i++) {
			if (group.get(i)) {
				count++;
			}
		}

		int[] list = new int[count];
		count = 0;
		for (int i = 0; i < m_numAttribs; i++) {
			if (group.get(i)) {
				list[count++] = i;
			}
		}
		return list;
	}

	/**
	 * reset to default values for options
	 */
	private void resetOptions() {
		m_population = null;
		m_popSize = 80;
		m_pCrossover = 1;
		m_pMutation = 0.033;
		m_maxGenerations = 10;
		m_reportFrequency = m_maxGenerations;
		m_starting = null;
		m_startRange = new Range();
		m_seed = 1;

		mode_remove_repetitive_pop=true;
		debug_print_pop=false;
	}

	/**
	 * by XieNaoban
	 * 
	 */
	
	/** Set Debug Mode */
	public boolean isPrintPop(){
		return debug_print_pop;
	}
	
	public void setPrintPop(boolean b){
		debug_print_pop = b;
	}
	
	protected void printPop(GABitSet[] m,int num) {
		int rk = 0;
		DecimalFormat df = new DecimalFormat("#######.##");
		System.out.println("@population"+num);
		for (int i = 0; i != m.length; ++i) {
			if (m[i].rank != rk) {
				rk = m[i].rank;
				System.out.println();
			}
			System.out.print("{" + df.format(m[i].getObjective().d[0]) + "," + df.format(m[i].getObjective().d[1]) + "} ");
		}
		System.out.println("\n");
	}

	/** Set whether remove repetitive pop */
	public boolean isRemoveRepetitivePop(){
		return mode_remove_repetitive_pop;
	}

	public void setRemoveRepetitivePop(boolean b){
		mode_remove_repetitive_pop = b;
	}
	
	protected void removeRepetitivePop() {
		Set<GABitSet> noRepet = new TreeSet<GABitSet>(new BitSetComparator());
		for (GABitSet e : m_population) {
			noRepet.add(e);
		}
		m_population = noRepet.toArray(new GABitSet[0]);
	}

	/** Comparators */
	protected class BitSetComparator implements Comparator<GABitSet> {
		public int compare(GABitSet gaarg1, GABitSet gaarg2) {
			BitSet arg1 = gaarg1.getChromosome();
			BitSet arg2 = gaarg2.getChromosome();
			if (arg1 == null && arg2 == null)
				return 0;
			if (arg1 == null)
				return 1;
			if (arg2 == null)
				return -1;
			if (arg1.equals(arg2))
				return 0;
			long[] l1 = arg1.toLongArray();
			long[] l2 = arg2.toLongArray();
			int len = (l1.length < l2.length ? l1.length : l2.length);
			for (int i = 0; i != len; ++i) {
				if (l1[i] > l2[i])
					return 1;
				if (l1[i] < l2[i])
					return -1;
			}
			if (l1.length < l2.length)
				return -1;
			return 1;
		}

	}

	protected class PopComparator implements Comparator<GABitSet> {
		int obj;

		public PopComparator(int n) {
			obj = n;
		}

		public int compare(GABitSet arg1, GABitSet arg2) {
			return (int) (arg2.getObjective().d[obj] - arg1.getObjective().d[obj]);
		}
	}

	protected class DPopComparator implements Comparator<GABitSet> {
		public int compare(GABitSet arg1, GABitSet arg2) {
			if (arg1 == null && arg2 == null)
				return 0;
			if (arg1 == null)
				return 1;
			if (arg2 == null)
				return -1;
			if (arg2.d > arg1.d)
				return 1;
			if (arg2.d < arg1.d)
				return -1;
			return 0;
		}

	}

	protected class IntComparator implements Comparator<int[]> {
		public int compare(int[] o1, int[] o2) {
			if (o1.length != o2.length)
				return o1.length - o2.length;
			for (int i = 0; i != o1.length; ++i)
				if (o1[i] != o2[i])
					return o1[i] - o2[i];
			return 0;
		}

	}

	/** Public functions */
	protected int dominate(MyDoub arg1, MyDoub arg2) {
		int flg;
		if (arg1.d[0] == arg2.d[0])
			flg = 0;
		else
			flg = (arg1.d[0] > arg2.d[0] ? 1 : -1);
		for (int i = 1; i < m_objects; ++i) {
			int flg2;
			if (arg1.d[i] == arg2.d[i])
				flg2 = 0;
			else
				flg2 = (arg1.d[i] > arg2.d[i] ? 1 : -1);
			if (flg == 0)
				flg = flg2;
			else if (flg2 == 0)
				continue;
			else if (flg != flg2)
				return 0; // no one dominates no one
		}
		return flg;
	}

	protected void nonDominatedSort() {
		/** set the s and n of each popu */
		for (int i = 0; i < m_population.length; ++i) {
			for (int j = i + 1; j < m_population.length; ++j) {
				int isDomi = dominate(m_population[i].getObjective(), m_population[j].getObjective());
				if (isDomi == 1) {
					m_population[i].sArr[m_population[i].s++] = m_population[j];
					++m_population[j].n;
				} else if (isDomi == -1) {
					++m_population[i].n;
					m_population[j].sArr[m_population[j].s++] = m_population[i];
				}
			}
		}

		/** set layers (the first layer) */
		Vector<Vector<GABitSet>> layers = new Vector<Vector<GABitSet>>();
		Vector<GABitSet> H = new Vector<GABitSet>();
		int layer = 0;
		for (int i = 0; i < m_population.length; ++i) {
			if (m_population[i].n == 0) {
				m_population[i].rank = layer;
				H.add(m_population[i]);
			}
		}
		layers.add(H);

		/** set layers (the other layers) */
		while (layers.get(layer).size() != 0) {
			H = new Vector<GABitSet>();
			for (GABitSet e : layers.get(layer)) {
				for (int j = 0; j < e.s; ++j)
					if (--e.sArr[j].n == 0) {
						e.sArr[j].rank = layer + 1;
						H.add(e.sArr[j]);
					}
			}
			if (H.size() > 0)
				layers.add(H);
			else
				break;
			++layer;
		}

		/** get new generation form lower layers */
		GABitSet[] newPop = new GABitSet[m_popSize];
		int sz = 0;
		Vector<GABitSet> last = null;
		for (Vector<GABitSet> ee : layers) {
			if (ee.size() + sz < m_popSize) {
				for (GABitSet e : ee) {
					newPop[sz++] = e.clone();
				}
			} else {
				last = ee;
				break;
			}
		}

		/** get degree of congestion */
		if (last == null) {
		} else {
			for (int i = 0; i != m_objects; ++i) {
				PopComparator cmp = new PopComparator(i);
				last.sort(cmp);
				for (int j = 1; j < last.size() - 1; ++j) {
					last.get(j).d += (last.get(j + 1).getObjective().d[i] - last.get(j - 1).getObjective().d[i]);
				}
			}
			DPopComparator cmpD = new DPopComparator();
			last.sort(cmpD);
			for (GABitSet e : last) {
				newPop[sz++] = e.clone();
				if (sz == m_popSize)
					break;
			}
		}
		m_population = newPop;
	}

	public int[][] search(WrapperSubsetEval ASEval, Instances data, String[] objs) throws Exception {
		System.out.println("@SEE2.timeUsed= -1");
		return null;
	}

	@Override
	public int[] search(ASEvaluation ASEvaluator, Instances data) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

}
