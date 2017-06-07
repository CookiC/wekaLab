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
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import weka.attributeSelection.*;
import weka.core.Instances;
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
public class NSGAIIS extends NSGAII {

	/** for serialization */
	static final long serialVersionUID = -1618264232838472679L;

	private Random m_random;
	
	/**
	 * Constructor. Make a new GeneticSearch object
	 */
	public NSGAIIS() {
		super();
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
	 * counts the number of features in a subset
	 * 
	 * @param featureSet
	 *            the feature set for which to count the features
	 * @return the number of features in the subset
	 */
	private int countFeatures(BitSet featureSet) {
		int count = 0;
		for (int i = 0; i < m_numAttribs; i++) {
			if (featureSet.get(i)) {
				count++;
			}
		}
		return count;
	}

	/**
	 * performs a single generation---selection, crossover, and mutation
	 * 
	 * @throws Exception
	 *             if an error occurs
	 */
	private void generation() throws Exception {
		int i, j = 0;
		MyDoub best_fit = new MyDoub();
		best_fit.set(-Double.MAX_VALUE);
		int old_count = 0;
		int count;
		GABitSet[] newPop = new GABitSet[m_popSize * 2];
		int parent1, parent2;
		List<GABitSet> parents = Arrays.asList(m_population);
		Collections.shuffle(parents);
		m_population=parents.toArray(new GABitSet[0]);
		/**
		 * first ensure that the population best is propogated into the new
		 * generation
		 */
		/*
		for (i = 0; i < m_popSize; i++) {
			if (m_population[i].getObjective().biggerThan(best_fit)) {
				j = i;
				best_fit = m_population[i].getObjective();
				old_count = countFeatures(m_population[i].getChromosome());
			} else if (m_population[i].getObjective().equalTo(best_fit)) {
				count = countFeatures(m_population[i].getChromosome());
				if (count < old_count) {
					j = i;
					best_fit = m_population[i].getObjective();
					old_count = count;
				}
			}
		}
		newPop[0] = (GABitSet) (m_population[j].clone());
		newPop[1] = newPop[0].clone();
		*/
		
		for (j = 0; j < m_popSize; j += 2) {
			do {
				parent1 = j;
				parent2 = j+1;
				newPop[j] = (GABitSet) (m_population[parent1].clone());
				newPop[j + 1] = (GABitSet) (m_population[parent2].clone());
				// if parents are equal mutate one bit
				if (parent1 == parent2) {
					int r;
					if (m_hasClass) {
						while ((r = m_random.nextInt(m_numAttribs)) == m_classIndex) {
							;
						}
					} else {
						r = m_random.nextInt(m_numAttribs);
					}

					if (newPop[j].get(r)) {
						newPop[j].clear(r);
					} else {
						newPop[j].set(r);
					}
				} else {
					// crossover
					double r = m_random.nextDouble();
					if (m_numAttribs >= 3) {
						if (r < m_pCrossover) {
							// cross point
							int cp = Math.abs(m_random.nextInt());

							cp %= (m_numAttribs - 2);
							cp++;

							for (i = 0; i < cp; i++) {
								if (m_population[parent1].get(i)) {
									newPop[j + 1].set(i);
								} else {
									newPop[j + 1].clear(i);
								}
								if (m_population[parent2].get(i)) {
									newPop[j].set(i);
								} else {
									newPop[j].clear(i);
								}
							}
						}
					}

					// mutate
					for (int k = 0; k < 2; k++) {
						for (i = 0; i < m_numAttribs; i++) {
							r = m_random.nextDouble();
							if (r < m_pMutation) {
								if (m_hasClass && (i == m_classIndex)) {
									// ignore class attribute
								} else {
									if (newPop[j + k].get(i)) {
										newPop[j + k].clear(i);
									} else {
										newPop[j + k].set(i);
									}
								}
							}
						}
					}

				}
			} while (newPop[j].getChromosome().length() == 0 || newPop[j + 1].getChromosome().length() == 0);
		}

		for (int k = 0; k != m_popSize; ++k)
			newPop[k + m_popSize] = (GABitSet) (m_population[k].clone());
		m_population = newPop;
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
	private void evaluatePopulation(WrapperSubsetEval ASEvaluator) throws Exception {
		int i;
		MyDoub merit = new MyDoub();
		for (i = 0; i < m_population.length; i++) {
			if (m_lookupTable.containsKey(m_population[i].getChromosome()) == false) {
				merit.d = ASEvaluator.evaluateSubset(m_population[i].getChromosome(), m_stateName);
				m_population[i].setObjective(merit);
				m_lookupTable.put(m_population[i].getChromosome(), m_population[i]);
			} else {
				GABitSet temp = m_lookupTable.get(m_population[i].getChromosome());
				m_population[i].setObjective(temp.getObjective());
			}
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
	private void initPopulation() throws Exception {
		int j, bit;
		int num_bits;
		boolean ok;
		
		Set<GABitSet> noRepet = new TreeSet<GABitSet>(new BitSetComparator());
		while(noRepet.size()<m_popSize) {
			GABitSet now = new GABitSet();
			num_bits = m_random.nextInt();
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
					bit = m_random.nextInt();
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
			noRepet.add(now);
		}
		m_population = noRepet.toArray(new GABitSet[0]);
	}

	@Override
	public int[][] search(WrapperSubsetEval ASEval, Instances data, String[] objs) throws Exception {
		long startMili=System.currentTimeMillis();
		m_stateName = objs;
		m_objects = objs.length;
		m_generationReports = new StringBuffer();

		m_hasClass = true;
		m_classIndex = data.classIndex();

		WrapperSubsetEval ASEvaluator = ASEval;
		m_numAttribs = data.numAttributes();

		m_startRange.setUpper(m_numAttribs - 1);
		if (!(getStartSet().equals(""))) {
			m_starting = m_startRange.getSelection();
		}

		// initial random population
		m_lookupTable = new Hashtable<BitSet, GABitSet>(m_lookupTableSize);
		m_random = new Random(m_seed);

		// set up random initial population
		initPopulation();
		evaluatePopulation(ASEvaluator);
		nonDominatedSort();
		
		if(isPrintPop()) printPop(m_population,0);
		
		for (int i = 1; i <= m_maxGenerations; i++) {
			List<GABitSet> parents = Arrays.asList(m_population);
			Collections.shuffle(parents);
			m_population=parents.toArray(new GABitSet[0]);
			generation();
			evaluatePopulation(ASEvaluator);
			if(isRemoveRepetitivePop()) removeRepetitivePop();
			nonDominatedSort();
			
			if(isPrintPop()) printPop(m_population,i);
		}

		int[][] ans;
		Set<int[]> ansSet = new TreeSet<int[]>(new IntComparator());
		for (GABitSet e : m_population) {
			if (e.rank != 0)
				break;
			ansSet.add(attributeList(e.getChromosome()));
		}
		
		int num = ansSet.size();
		ans = new int[num][];
		int iii=0;
		for (int[] e : ansSet) {
			ans[iii++] = e;
		}
		long endMili=System.currentTimeMillis();
		System.out.println("@NSGAIIS.timeUsed="+(endMili-startMili));
		return ans;
	}

}
