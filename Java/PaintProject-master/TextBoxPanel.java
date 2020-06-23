package ca.utoronto.utm.paint;

import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.util.Observable;
import java.util.Observer;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.event.ChangeListener;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;


public class TextBoxPanel extends JPanel implements FocusListener, DocumentListener{

	JTextField textField = new JTextField(20);
	private TextBoxPanelTalksToPaintModel Talker = new TextBoxPanelTalksToPaintModel();
	private boolean focus = false;
	
	public TextBoxPanel(ChangeListener l) {
		this.setLayout(new FlowLayout());
		
		//button.addActionListener(new ActionListener() { 
			//  public void actionPerformed(ActionEvent e) { 
				//   text = textField.getText();
				 // } 
				//} );
		textField.setText("Insert your Text Here");
		textField.getDocument().addDocumentListener(this);
		textField.addFocusListener(this);
		
		this.add(textField);
		
	}

	@Override
	public void focusGained(FocusEvent arg0) { 
		if(this.textField.getText().trim().equals("Insert your Text Here"))
			this.textField.setText("");
		else {}
		this.focus = true;
        this.Talker.setText(this.textField.getText());
	}

	@Override
	public void focusLost(FocusEvent arg0) {
		this.focus = false;
        if(this.textField.getText().trim().equals(""))
        	this.textField.setText("Insert your Text Here");
        else {}
        this.Talker.setText(this.textField.getText());
	}

	@Override
	public void changedUpdate(DocumentEvent arg0) {
		
	}

	@Override
	public void insertUpdate(DocumentEvent arg0) {
		if(focus) {
	        this.Talker.setText(this.textField.getText());
		}
		
	}

	@Override
	public void removeUpdate(DocumentEvent arg0) {
		if(focus) {
	        this.Talker.setText(this.textField.getText());
		}
		
	}

	public void addTBPTTPM(TextBoxPanelTalksToPaintModel T) {
		this.Talker = T;
		
	}
}
