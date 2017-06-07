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

import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.Hashtable;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import weka.attributeSelection.*;
import weka.core.Instances;

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
public class NSGAIIP extends NSGAII {

	/** for serialization */
	static final long serialVersionUID = -1618264232838472679L;

	/**
	 * holds a starting set as an array of attributes. Becomes one member of the
	 * initial random population
	 */

	/** number of threads */
	private int m_threadsNum;
	
	/** threads */
	private EachThread[] m_threads;
	
	/** offer pop to threads */
	private LinkedList<GABitSet> m_popQue;

	/** random number generation */
	private Random[] m_random;

	/** used in threads to count how many pops have been calculated */
	Integer m_sumPop;
	
	/** locks */
	private Object m_lock;
	private Object m_sumPopLock;

	/**
	 * Constructor. Make a new GeneticSearch object
	 */
	public NSGAIIP() {
		super();
		m_threadsNum = 4;
	}

	/**
	 * performs a single generation---selection, crossover, and mutation
	 * 
	 * @throws Exception
	 *             if an error occurs
	 */
	private void generation(GABitSet parent0,GABitSet parent1,Random rand) throws Exception {
		BitSet[] now=new BitSet[2];
		
		do {
			now[0] = (BitSet)parent0.getChromosome().clone();
			now[1] = (BitSet)parent1.getChromosome().clone();
			
			// crossover
			double r = rand.nextDouble();
			if (m_numAttribs >= 3) {
				if (r < m_pCrossover) {
					// cross point
					int cp = Math.abs(rand.nextInt());

					cp %= (m_numAttribs - 2);
					cp++;

					for (int i = 0; i < cp; i++) {
						if (parent0.get(i)) {
							now[1].set(i);
						} else {
							now[1].clear(i);
						}
						if (parent1.get(i)) {
							now[0].set(i);
						} else {
							now[0].clear(i);
						}
					}
				}
			}

			// mutate
			for (int k = 0; k < 2; k++) {
				for (int i = 0; i < m_numAttribs; i++) {
					r = rand.nextDouble();
					if (r < m_pMutation) {
						if (m_hasClass && (i == m_classIndex)) {
							// ignore class attribute
						} else {
							if (now[k].get(i)) {
								now[k].clear(i);
							} else {
								now[k].set(i);
							}
						}
					}
				}
			}
		} while (now[0].length() == 0 || now[1].length() == 0);
		
		parent0.setChromosome(now[0]);
		parent1.setChromosome(now[1]);
	}

	/**
	 * evaluates an entire population. Population members are looked up in a
	 * hash table and if they are not found then they are evaluated using
	 * ASEvaluator.
	 * 
	 * @param ASEvaluator
	 *            the subset evaluator to use for evaluating population members
	 * @throws Exception
	 *             if something goes wrong during evaluation
	 */
	private void evaluatePopulation(GABitSet now, WrapperSubsetEval ASEvaluator) throws Exception {
		MyDoub merit = new MyDoub();
		if (m_lookupTable.containsKey(now.getChromosome()) == false) {
				merit.d = ASEvaluator.evaluateSubset(now.getChromosome(), m_stateName);
			now.setObjective(merit);
			m_lookupTable.put(now.getChromosome(), now);
		} else {
			GABitSet temp = m_lookupTable.get(now.getChromosome());
			now.setObjective(temp.getObjective());
		}
	}

	/**
	 * creates random population members for the initial population. Also sets
	 * the first population member to be a start set (if any) provided by the
	 * user
	 * 
	 * @throws Exception
	 *             if the population can't be created
	 */
	private void initPopulation(GABitSet now,Random rand) throws Exception {
		int j, bit;
		int num_bits;
		boolean ok;

		num_bits = rand.nextInt();
		num_bits = num_bits % m_numAttribs - 1;
		if (num_bits < 0) {
			num_bits *= -1;
		}
		if (num_bits == 0) {
			num_bits = 1;
		}

		for (j = 0; j < num_bits; j++) {
			ok = false;
			do {
				bit = rand.nextInt();
				if (bit < 0) {
					bit *= -1;
				}
				bit = bit % m_numAttribs;
				if (m_hasClass) {
					if (bit != m_classIndex) {
						ok = true;
					}
				} else {
					ok = true;
				}
			} while (!ok);

			if (bit > m_numAttribs) {
				throw new Exception("Problem in population init");
			}
			now.set(bit);
		}
	}

	/**
	 * by XieNaoban
	 * 
	 */

	public void setThreadsNum(int n){
		m_threadsNum = n;
	}
	
	private class InitThread extends Thread
	{
		public int num;
		public Set<GABitSet> noRepet;
		Random rand;
		WrapperSubsetEval evaluator;
		
		public InitThread(int n, Set<GABitSet> nr,Random r,WrapperSubsetEval ASE)
		{
			num=n;
			noRepet=nr;
			rand = r;
			evaluator=ASE;
		}
		
		public void run()
		{
			GABitSet now;
			while (true) {
				now = new GABitSet();
				try {
					initPopulation(now,rand);
					evaluatePopulation(now, evaluator);
					synchronized (noRepet) {
						noRepet.add(now);
						if(noRepet.size()>m_popSize) return;
					}
				} catch (Exception e) {
					System.out.println("init error in threads: " + e);
				}
			}
		}
	}
	
	void calInitGene()
	{
		Set<GABitSet> noRepet = new TreeSet<GABitSet>(new BitSetComparator());
		InitThread[] thds = new InitThread[m_threadsNum];
		for (int th = 0; th < m_threadsNum; ++th) {
			thds[th] = new InitThread(th, noRepet, m_random[th], ASEvaluator[th]);
			thds[th].start();
		}
		for (int th = 0; th < m_threadsNum; ++th) {
			try {
				thds[th].join();
			} catch (Exception e) {
				System.out.println("error in join: " + e);
			}
		}
		m_population = noRepet.toArray(new GABitSet[0]);
		nonDominatedSort();
	}

	private GABitSet[] takeTwoPop() {
		GABitSet[] ans = new GABitSet[2];
		synchronized (m_popQue) {
			try {
				while (m_popQue.size() == 0)
					m_popQue.wait();	// Changed to "m_popQue.size() < 2" later
				ans[0] = m_popQue.remove();
				ans[1] = m_popQue.remove();
				m_popQue.notify();
			} catch (Exception e) {
				System.out.println("Errors in takeTwoPop(): " + e);
			}
		}
		return ans;
	}

	private class EachThread extends Thread {
		public int num;
		
		public Random rand;
		public WrapperSubsetEval evaluator;
		
		public EachThread(int n, Random r, WrapperSubsetEval ASE) {
			num = n;
			rand = r;
			evaluator = ASE;
		}

		public void run() {
			GABitSet[] now;
			while (true) {
				try {
					now = takeTwoPop();
					generation(now[0], now[1], rand);
					evaluatePopulation(now[0], evaluator);
					evaluatePopulation(now[1], evaluator);
				} catch (Exception e) {
					System.out.println("Errors in EachThread.run(): " + e);
				}
				synchronized (m_sumPopLock) {
					if ((m_sumPop += 2) >= m_popSize) {
						synchronized (m_lock) {
							m_lock.notify();
						}
					}
				}
			}

		}
	}

	@Override
	public int[][] search(WrapperSubsetEval ASEval, Instances data, String[] objs) throws Exception {
		long startMili = System.currentTimeMillis();
		m_stateName = objs;
		m_objects = objs.length;
		m_generationReports = new StringBuffer();
		m_lock = new Object();
		m_sumPopLock = new Object();

		m_hasClass = true;
		m_classIndex = data.classIndex();

		ASEvaluator = new WrapperSubsetEval[m_threadsNum];
		ASEvaluator[0] = ASEval;
		for(int i=1;i<m_threadsNum;++i){
			ASEvaluator[i] = (WrapperSubsetEval)ASEval.clone();
		}
		m_numAttribs = data.numAttributes();

		m_startRange.setUpper(m_numAttribs - 1);
		if (!(getStartSet().equals(""))) {
			m_starting = m_startRange.getSelection();
		}

		m_popQue = new LinkedList<GABitSet>();
		m_threads = new EachThread[m_threadsNum];

		m_random = new Random[m_threadsNum];
		for (int i = 0; i < m_threadsNum; ++i)
			m_random[i] = new Random(m_seed + i);

		m_lookupTable = new Hashtable<BitSet, GABitSet>(m_lookupTableSize);

		// Set up random initial population
		calInitGene();
		if (isPrintPop())
			printPop(m_population, 0);

		// The other generations
		m_sumPop = 0;
		for (int th = 0; th < m_threadsNum; ++th) {
			m_threads[th] = new EachThread(th, m_random[th], ASEvaluator[th]);
			m_threads[th].setDaemon(true);
			m_threads[th].start();
		}

		for (int i = 1; i <= m_maxGenerations; i++) {
			m_sumPop = 0;
			GABitSet[] newPop = new GABitSet[2 * m_popSize];
			for (int j = 0; j < m_popSize; ++j) {
				newPop[j] = m_population[j];
				m_population[j] = newPop[j].clone();
			}
			List<GABitSet> parents = Arrays.asList(m_population);
			Collections.shuffle(parents);
			synchronized (m_lock) {
				synchronized (m_popQue) {
					m_popQue.addAll(parents);
					m_popQue.notify();
				}
				m_lock.wait();
			}
			for (int j = 0; j < m_popSize; ++j) {
				newPop[j + m_popSize] = parents.get(j);
			}
			m_population = newPop;
			if (isRemoveRepetitivePop())
				removeRepetitivePop();
			nonDominatedSort();
			if (isPrintPop())
				printPop(m_population, i);
		}

		// for (int th = 0; th < m_numThreads; ++th) m_threads[th].stop();

		int[][] ans;
		Set<int[]> ansSet = new TreeSet<int[]>(new IntComparator());
		for (GABitSet e : m_population) {
			if (e.rank != 0)
				break;
			ansSet.add(attributeList(e.getChromosome()));
		}
		int num = ansSet.size();
		ans = new int[num][];
		for (int[] e : ansSet) {
			ans[--num] = e;
		}
		long endMili = System.currentTimeMillis();
		System.out.println("@NSGAIIP.timeUsed=" + (endMili - startMili));
		return ans;
	}
	
	void DEBUG(String s){
		System.out.println("Debug: " + s);
	}
}
