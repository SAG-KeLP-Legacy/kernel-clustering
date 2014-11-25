/*
 * Copyright 2014 Simone Filice and Giuseppe Castellucci and Danilo Croce and Roberto Basili
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

package it.uniroma2.sag.kelp.learningalgorithm.clustering.kernelbasedkmeans;

import com.fasterxml.jackson.annotation.JsonTypeName;

import it.uniroma2.sag.kelp.data.clustering.ClusterExample;
import it.uniroma2.sag.kelp.data.example.Example;

@JsonTypeName("kernelbasedkmeansexample")
public class KernelBasedKMeansExample extends ClusterExample {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5368757832244686390L;

	public KernelBasedKMeansExample() {
	}

	public KernelBasedKMeansExample(Example e, float dist) {
		this();
		this.example = e;
		this.dist = dist;
	}

	private Example example;
	private Float dist;

	@Override
	public int compareTo(ClusterExample arg0) {
		int res = dist.compareTo(((KernelBasedKMeansExample) arg0).dist);
		if (res == 0)
			return example.toString().compareTo(arg0.toString());
		return res;
	}

	@Override
	public Example getExample() {
		return example;
	}

	public void setExample(Example example) {
		this.example = example;
	}

	public Float getDist() {
		return dist;
	}

	public void setDist(Float dist) {
		this.dist = dist;
	}

	@Override
	public String toString() {
		return dist + "\t" + example;
	}

}
