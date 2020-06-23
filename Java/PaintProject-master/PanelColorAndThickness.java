package ca.utoronto.utm.paint;

import java.awt.Color;
import java.awt.GridLayout;

import javax.swing.JFrame;
import javax.swing.JPanel;

public class PanelColorAndThickness extends JPanel {
	
	
	DataColorPanel data= new DataColorPanel();
	ColorPanel p = new ColorPanel(data);
	ThicknessListenerAndModel tl = new ThicknessListenerAndModel();;
	ThicknessPanel thickness = new ThicknessPanel(tl);
	TextBoxPanelTalksToPaintModel l = new TextBoxPanelTalksToPaintModel();
	TextBoxPanel text = new TextBoxPanel(l);
	public PanelColorAndThickness() {
	
	this.setLayout(new GridLayout(1,3));
	this.setVisible(true);

	data.addObserver(p);
	this.add(p);
	

	tl.addObserver(thickness);
	this.add(thickness);
	
	text.addTBPTTPM(l);
	this.add(text);

}


public DataColorPanel getColorData()
{
return data;
}

public ThicknessListenerAndModel getThicknessModel(){
	return tl;
}

public TextBoxPanelTalksToPaintModel getTextBoxModel(){
	return l;
}

/**
public static void main(String[] args) {
	//test use only

	javax.swing.SwingUtilities.invokeLater(new Runnable() {
		public void run() {
			PanelColorAndThickness f = new PanelColorAndThickness();
		
		}
	});

}
**/
}
