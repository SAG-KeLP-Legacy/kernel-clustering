/*
 * Copyright 2015 Simone Filice and Giuseppe Castellucci and Danilo Croce and Roberto Basili
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.uniroma2.sag.kelp.learningalgorithm.clustering.kmeans;

import it.uniroma2.sag.kelp.data.clustering.Cluster;
import it.uniroma2.sag.kelp.data.clustering.ClusterExample;
import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.representation.Vector;
import it.uniroma2.sag.kelp.learningalgorithm.clustering.ClusteringAlgorithm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.TreeMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonTypeName;

/**
 * Implements the K-means Clustering Algorithm, that works on an Explicit
 * feature Space. It differs from the
 * <code>it.uniroma2.sag.kelp.learningalgorithm.clustering.kernelbasedkmeans.KernelBasedKMeansEngine</code>
 * as, at each iteration, it explicitly computes the centroid of each cluster.
 * 
 * The computational cost is thus O(Ikn), where: <br>
 * - I: is the number of iterations <br>
 * - k: the number of desired clusters<br>
 * - n: the number of input examples <br>
 * <br>
 * while the complexity of
 * <code>it.uniroma2.sag.kelp.learningalgorithm.clustering.kernelbasedkmeans.KernelBasedKMeansEngine</code>
 * is O(n^2).
 * 
 * For more details on the K-mean Algorithm, please refer to:
 * <code>https://en.wikipedia.org/wiki/K-means_clustering</code>
 * 
 * @author Danilo Croce
 */
@JsonTypeName("kmeans")
public class LinearKMeansEngine implements ClusteringAlgorithm {
	private Logger logger = LoggerFactory.getLogger(LinearKMeansEngine.class);

	/**
	 * The number of expected clusters
	 */
	private int k;

	/**
	 * The maximum number of iterations
	 */
	private int maxIterations;

	/**
	 * The representaion name
	 */
	private String representationName;

	public LinearKMeansEngine() {

	}

	/**
	 * @param representationName
	 *            The representation name containing the vector used by the
	 *            algorithm
	 * @param k
	 *            The number of expected clusters
	 * @param maxIterations
	 *            The maximum number of iterations
	 */
	public LinearKMeansEngine(String representationName, int k,
			int maxIterations) {
		this.representationName = representationName;
		this.k = k;
		this.maxIterations = maxIterations;
	}

	/**
	 * Measure the euclidean distance between an example a the centroid of a
	 * cluster
	 * 
	 * @param example
	 * @param cluster
	 * @return
	 */
	private float calculateDistance(Example example, LinearKMeansCluster cluster) {

		Vector exampleVector = (Vector) example
				.getRepresentation(representationName);

		if (cluster.getCentroid() == null) {
			logger.warn("Waning:\t Centroid is null");
			return exampleVector.getSquaredNorm();
		}

		Vector centroid = cluster.getCentroid();

		return centroid.euclideanDistance(exampleVector);
	}

	@Override
	public List<Cluster> cluster(Dataset dataset) {
		/*
		 * Check consistency: the number of input examples MUST be greater or
		 * equal to the target K
		 */
		if (dataset.getNumberOfExamples() < k) {
			System.err.println("Error: the number of instances ("
					+ dataset.getNumberOfExamples()
					+ ") must be higher than k (" + k + ")");
			return null;
		}

		/*
		 * Initialize seed and outputStructures
		 */
		List<LinearKMeansCluster> resClusters = new ArrayList<LinearKMeansCluster>();
		ArrayList<Example> seedVector = getFirstExamplesAsSeed(dataset
				.getExamples());
		for (int clusterId = 0; clusterId < k; clusterId++) {
			resClusters.add(new LinearKMeansCluster("cluster_" + clusterId));
			if (clusterId < seedVector.size()) {
				LinearKMeansExample linearKMeansExample = new LinearKMeansExample(
						seedVector.get(clusterId), 0);

				resClusters.get(clusterId).add(linearKMeansExample);
				resClusters.get(clusterId).updateCentroid(representationName);
			}
		}

		/*
		 * Do Work
		 */
		// For each iteration
		for (int t = 0; t < maxIterations; t++) {

			int reassignment;

			logger.debug("\nITERATION:\t" + (t + 1));

			TreeMap<Long, Integer> exampleIdToClusterMap = new TreeMap<Long, Integer>();

			HashMap<Example, Float> minDistances = new HashMap<Example, Float>();

			/*
			 * Searching for the nearest cluster
			 */
			for (Example example : dataset.getExamples()) {

				float minValue = Float.MAX_VALUE;
				int targetCluster = -1;

				for (int clusterId = 0; clusterId < k; clusterId++) {

					float d = calculateDistance(example,
							(LinearKMeansCluster) resClusters.get(clusterId));

					logger.debug("Distance of " + example.getId()
							+ " from cluster " + clusterId + ":\t" + d);

					if (d < minValue) {
						minValue = d;
						targetCluster = clusterId;
					}
				}

				minDistances.put(example, minValue);
				exampleIdToClusterMap.put(example.getId(), targetCluster);
			}

			/*
			 * Counting reassignments
			 */
			reassignment = countReassigment(exampleIdToClusterMap, resClusters);

			logger.info("Reassigments:\t" + reassignment);

			/*
			 * Updating
			 */
			for (int i = 0; i < resClusters.size(); i++)
				resClusters.get(i).clear();

			for (Example example : dataset.getExamples()) {
				logger.debug("Re-assigning " + example.getId() + " to "
						+ exampleIdToClusterMap.get(example.getId()));

				int assignedClusterId = exampleIdToClusterMap.get(example
						.getId());
				float minDist = minDistances.get(example);

				LinearKMeansExample linearKMeansExample = new LinearKMeansExample(
						example, minDist);

				resClusters.get(assignedClusterId).add(linearKMeansExample);
			}

			for (int i = 0; i < resClusters.size(); i++)
				resClusters.get(i).updateCentroid(representationName);

			if (t > 0 && reassignment == 0) {
				break;
			}
		}

		/*
		 * Sort results by distance from the controid.
		 */
		for (Cluster c : resClusters) {
			c.sortAscendingOrder();
		}

		ArrayList<Cluster> res = new ArrayList<Cluster>();
		for (LinearKMeansCluster linearKMeansCluster : resClusters) {
			res.add(linearKMeansCluster);
		}

		return res;

	}

	/**
	 * Count the reassignment as a stopping criteria for the algorithm
	 * 
	 * @param exampleIdToClusterMap
	 *            The map of assignment for the previous iteration
	 * @param clusterList
	 *            The actual clusters
	 * @return
	 */
	private int countReassigment(TreeMap<Long, Integer> exampleIdToClusterMap,
			List<LinearKMeansCluster> clusterList) {

		int reassignment = 0;

		TreeMap<Long, Integer> currentExampleIdToClusterMap = new TreeMap<Long, Integer>();

		int clusterId = 0;
		for (Cluster cluster : clusterList) {
			for (ClusterExample clusterExample : cluster.getExamples()) {
				currentExampleIdToClusterMap.put(clusterExample.getExample()
						.getId(), clusterId);
			}
			clusterId++;
		}

		for (Long currentExId : currentExampleIdToClusterMap.keySet()) {
			if (exampleIdToClusterMap.get(currentExId).intValue() != currentExampleIdToClusterMap
					.get(currentExId).intValue())
				reassignment++;
		}

		return reassignment;
	}

	/**
	 * Select the seeds by selecting the first examples
	 * 
	 * @param inputExamples
	 * @return The seed examples
	 */
	@JsonIgnore
	private ArrayList<Example> getFirstExamplesAsSeed(
			List<Example> inputExamples) {

		ArrayList<Example> seeds = new ArrayList<Example>();

		int addedSeedCounter = 0;

		for (Example ex : inputExamples) {

			if (++addedSeedCounter > k)
				break;

			seeds.add(ex);
		}

		int i = 0;
		for (Example ex : seeds) {
			logger.debug("Seed " + i++ + ": " + Arrays.toString(ex.getLabels()));
		}

		return seeds;
	}

	public int getK() {
		return k;
	}

	public int getMaxIterations() {
		return maxIterations;
	}

	public String getRepresentationName() {
		return representationName;
	}

	public void setK(int k) {
		this.k = k;
	}

	public void setMaxIterations(int maxIterations) {
		this.maxIterations = maxIterations;
	}

	public void setRepresentationName(String representationName) {
		this.representationName = representationName;
	}

}
