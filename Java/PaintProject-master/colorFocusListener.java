package ca.utoronto.utm.paint;

import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;

import javax.swing.JTextField;

public class colorFocusListener implements FocusListener{
	private String color = "";
	private JTextField textField = new JTextField("");
	
	public void setColor(String s) {
		this.color = s;
	}
	
	public String getColor() {
		return this.color;
	}
	
	public void addTextField(JTextField T) {
		this.textField = T;
	}
	
	public JTextField getTextField() {
		return this.textField;
	}
	
	@Override
	public void focusGained(FocusEvent arg0) { 
		if(this.textField.getText().trim().equals(this.color))
        this.textField.setText("");
     else {}
	}

	@Override
	public void focusLost(FocusEvent arg0) {
        if(this.textField.getText().trim().equals(""))
        	this.textField.setText(this.color);
         else {}
	}
}
