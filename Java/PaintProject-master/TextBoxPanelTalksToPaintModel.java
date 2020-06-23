package ca.utoronto.utm.paint;


import java.util.Observable;

import javax.swing.JButton;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

public class TextBoxPanelTalksToPaintModel  extends Observable implements ChangeListener{
	
	private String text;
	
	public TextBoxPanelTalksToPaintModel() {
		this.text = "";
	}
	
	public String getText() {
		return this.text;
	}
	
	public void setText(String s) {
		this.text = s;
		this.setChanged();
		this.notifyObservers(this.text);
	}

	@Override
	public void stateChanged(ChangeEvent e) {
		JButton source = (JButton)e.getSource();
		this.text = (String) source.getText();
		this.setChanged();
		this.notifyObservers(this.text);
	}

}
