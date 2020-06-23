package ca.utoronto.utm.paint;

import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import java.util.Observable;

import javax.swing.JSlider;

public class ThicknessListenerAndModel extends Observable implements ChangeListener{
	private int Thickness;
	
	public ThicknessListenerAndModel(){
		Thickness = 1;
	}
	public int getThickness(){
		return this.Thickness;
	}
	@Override
	public void stateChanged(ChangeEvent e) {
	    JSlider source = (JSlider)e.getSource();
		this.Thickness = (int) source.getValue();
		this.setChanged();		
		this.notifyObservers(this.Thickness);
	}

}

