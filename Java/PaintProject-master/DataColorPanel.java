package ca.utoronto.utm.paint;

import java.awt.Color;
import java.util.Observable;

public class DataColorPanel extends Observable {
	private Color c;
	public DataColorPanel(){
		c = new Color(0);
	}
	
	public Color getColor(){
		return c;
		
	}
	public void setColor(Color c){
		this.c=c;
		this.setChanged();
		this.notifyObservers(c);
	}
}
