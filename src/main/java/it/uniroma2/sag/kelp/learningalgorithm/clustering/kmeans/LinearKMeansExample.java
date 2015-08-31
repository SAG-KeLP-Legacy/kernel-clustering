package it.uniroma2.sag.kelp.learningalgorithm.clustering.kmeans;

import com.fasterxml.jackson.annotation.JsonTypeName;

import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.learningalgorithm.clustering.kernelbasedkmeans.KernelBasedKMeansExample;

@JsonTypeName("linearkmeansexample")
public class LinearKMeansExample extends KernelBasedKMeansExample {

	private static final long serialVersionUID = 4309082543662353543L;

	public LinearKMeansExample(Example e, float dist) {
		super(e, dist);
	}

	public LinearKMeansExample() {

	}

}
