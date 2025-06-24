#pragma once
#include <atomic> 
#include <objbase.h>
#include <string> 

using ULONG = unsigned long;

class GObject
{
private:
    static              std::atomic<GUID> s_nextId;

	

    GObject*            m_prev;               // previous item/parent of list
    GObject*            m_next;               // next item/child of list
	GObject*            m_cousin;             // cousin item of list, if any
	GObject*            m_sibling;            // sibling item of list, if any

    GUID                id;				      // unique identifier for the object

public:
                       
	                    GObject(void);
	virtual             ~GObject(void) = default;
    // Pure virtual function to get the type ID of the object.  Must be implemented by derived classes
	virtual             int GetTypeID() const = 0; //Type() method is replaced by GetTypeID() in derived classes

    //--- methods to access protected data
    UUID                Id() const;
    std::wstring        Name() const;
    GObject*            Prev(void) const;
    void                Prev(GObject* node);
    GObject*            Next(void) const;
    void                Next(GObject* node);
    //--- methods for working with files
    virtual bool        Save(const int file_handle) const;
    virtual bool        Save(HANDLE file_handle) const;
    virtual bool        Save(std::ofstream& outFile) const;
    virtual bool        Load(const int file_handle);
	virtual bool        Load(HANDLE file_handle);
    virtual bool        Load(std::ifstream& inFile);
    //--- method of identifying the object
    //OBSOLETE, use GetTypeID() instead
	virtual int         Type(void) const;
    //--- method of comparing the objects
    virtual int         Compare(const GObject* node, const int mode = 0) const;
};



