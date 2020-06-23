package ca.utoronto.utm.paint;

import java.awt.Color;
import java.awt.event.ActionListener;
import java.util.Observable;
import java.util.Observer;

import javax.swing.ImageIcon;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.JTextField;
import javax.swing.event.ChangeListener;

public class ThicknessPanel extends JPanel implements Observer{

private int Thickness;
JLabel fid = new JLabel();
ImageIcon icons[]= new ImageIcon[24];


public ThicknessPanel(ChangeListener listener){
	
	for(int i = 0; i<icons.length; i++) {
		icons[i] = new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/"+(i)+".PNG");
	}
	/*
	 * 
	icons[0]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/1.PNG");
	icons[1]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/2.PNG");
	icons[2]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/3.PNG");
	icons[3]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/4.PNG");
	icons[4]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/5.PNG");
	icons[5]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/6.PNG");
	icons[6]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/7.PNG");
	icons[7]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/8.PNG");
	icons[8]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/9.PNG");
	icons[9]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/10.PNG");
	icons[10]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/11.PNG");
	icons[11]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/12.PNG");
	icons[12]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/13.PNG");
	icons[13]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/14.PNG");
	icons[14]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/15.PNG");
	icons[15]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/16.PNG");
	icons[16]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/17.PNG");
	icons[17]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/18.PNG");
	icons[9]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/19.PNG");
	icons[1]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/20.PNG");
	icons[2]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/21.PNG");
	icons[3]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/22.PNG");
	icons[4]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/23.PNG");
	icons[5]=new ImageIcon("src/ca/utoronto/utm/paint/thicknessPNGs/24.PNG");
	 */

	int minThick = 0;
	int maxThick = 24;
	JSlider thickness = new JSlider(JSlider.HORIZONTAL,
            minThick, maxThick, 1);
	add(thickness);
	
	//Turn on labels at major tick marks.
	thickness.setMajorTickSpacing(6);
	thickness.setMinorTickSpacing(2);
	thickness.setPaintTicks(true);
	thickness.setPaintLabels(true);
	//JButton thickness = new JButton("Thickness");
	thickness.addChangeListener(listener);
	fid.setHorizontalAlignment(JTextField.CENTER);
	fid.setIcon(icons[1]);
	fid.setBackground(Color.white);

	thickness.setBounds(20, 8, 90, 38);
	fid.setBounds(120, 8, 180, 38);
	this.add(thickness);
	this.add(fid);
	this.setLayout(null);
}

@Override
public void update(Observable arg0, Object arg1) {
	this.Thickness = (int)arg1;
	if(Thickness == 24) Thickness = 23;
	fid.setIcon(icons[Thickness]);
}
}
