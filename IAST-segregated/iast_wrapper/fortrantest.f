      Program Testiast
      Implicit None

      Include 'commons.inc'
C     See backup folder 
      Integer J,I,Nterm_max,Carrier_gas
      
      Double Precision P,Yi(Maxcomp),Xi1(Maxcomp)
     $     ,Ni(Maxcomp),Molfrac(Maxcomp,Maxterm)
C     Note to self: program written in Fortran 2003. 
C     An Inert Component Should Have
C     Nterm=1
C     Langmuir=True
C     Ki=0
C     Nimax Not Equal To Zero

C     Start for Python
      Ncomp = 2
      Nterm_max = 2
C     Nterm_max means max no. of sides in pure-component isotherms. 

C     C7
      Ki(1,1)       = 0.529536d0
      Ki(1,2)       = 0.685660d0
      Nimax(1,1)    = 0.075591d0
      Nimax(1,2)    = 292.889202d0
      Pow(1,1)      = 1.0d0
      Pow(1,2)      = 1.0d0
      Langmuir(1,1) = .True.
      Langmuir(1,2) = .True.      

C     2mC6 
      Ki(2,1)       = 0.055617d0
      Ki(2,2)       = 0.000234d0
      Nimax(2,1)    = 0.698642d0
      Nimax(2,2)    = 36.891289d0
      Pow(2,1)      = 1.0d0
      Pow(2,2)      = 1.0d0
      Langmuir(2,1) = .True.
      Langmuir(2,2) = .True.
      


      Carrier_gas = 1       
      P   = 1.0d4
C     Mole fractions below
      Yi(1) = 0.50d0
      Yi(2) = 0.50d0      
C      Yi(3) = 0.50d0      

C     End for Python
      Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)

      write(6,*) 'Ni(2)   ','Ni(3)   '
      write(6,'(2e20.10)') Ni(2),Ni(3)

C      stop

      Do J = 0, 8
         If(J.Lt.8) Then
            Do I = 1, 3, 2
               P = Dble(I*10**J)
               Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)
               Write(25,'(5e20.10)') P,Ni(2),Ni(3)
            Enddo
         Else
            P = Dble(1.0*10**J)
            Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)
            Write(25,'(5e20.10)') P,Ni(2),Ni(3)
         Endif
      Enddo
      
      
      Stop
      End
      
